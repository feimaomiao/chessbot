import aiosqlite
from datetime import datetime, timedelta
from typing import Optional
from .models import Guild, TrackedPlayer, Game, ActiveQuiz


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Initialize database connection and create tables."""
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()

    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()

    async def _create_tables(self):
        """Create database tables if they don't exist."""
        await self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS guilds (
                id INTEGER PRIMARY KEY,
                notification_channel_id INTEGER,
                summary_channel_id INTEGER,
                summary_time TEXT DEFAULT '09:00'
            );

            CREATE TABLE IF NOT EXISTS tracked_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                platform TEXT NOT NULL,
                username TEXT NOT NULL,
                display_name TEXT,
                discord_user_id INTEGER,
                added_by INTEGER,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (guild_id) REFERENCES guilds(id),
                UNIQUE(guild_id, platform, username)
            );

            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                game_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                time_control TEXT,
                time_control_display TEXT,
                result TEXT,
                player_color TEXT,
                rating_after INTEGER,
                rating_change INTEGER,
                opponent TEXT,
                opponent_rating INTEGER,
                played_at DATETIME,
                game_url TEXT,
                final_fen TEXT,
                notified BOOLEAN DEFAULT 0,
                accuracy REAL,
                termination TEXT,
                FOREIGN KEY (player_id) REFERENCES tracked_players(id),
                UNIQUE(player_id, game_id)
            );

            CREATE INDEX IF NOT EXISTS idx_games_player_id ON games(player_id);
            CREATE INDEX IF NOT EXISTS idx_games_played_at ON games(played_at);
            CREATE INDEX IF NOT EXISTS idx_tracked_players_guild ON tracked_players(guild_id);

            CREATE TABLE IF NOT EXISTS active_quizzes (
                channel_id INTEGER PRIMARY KEY,
                guild_id INTEGER NOT NULL,
                position_fen TEXT NOT NULL,
                correct_move_san TEXT NOT NULL,
                played_move_san TEXT NOT NULL,
                game_url TEXT NOT NULL,
                player_username TEXT NOT NULL,
                opponent_username TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                difficulty TEXT NOT NULL,
                eval_before REAL NOT NULL,
                eval_after_best REAL NOT NULL,
                eval_after_played REAL NOT NULL,
                started_at DATETIME NOT NULL,
                FOREIGN KEY (guild_id) REFERENCES guilds(id)
            );

            CREATE TABLE IF NOT EXISTS quiz_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                score INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (guild_id) REFERENCES guilds(id),
                UNIQUE(guild_id, user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_quiz_scores_guild ON quiz_scores(guild_id);
        """)
        await self._connection.commit()

        # Migration: Add discord_user_id column if it doesn't exist
        try:
            await self._connection.execute(
                "ALTER TABLE tracked_players ADD COLUMN discord_user_id INTEGER"
            )
            await self._connection.commit()
        except Exception:
            pass  # Column already exists

        # Migration: Add accuracy column if it doesn't exist
        try:
            await self._connection.execute(
                "ALTER TABLE games ADD COLUMN accuracy REAL"
            )
            await self._connection.commit()
        except Exception:
            pass  # Column already exists

        # Migration: Add termination column if it doesn't exist
        try:
            await self._connection.execute(
                "ALTER TABLE games ADD COLUMN termination TEXT"
            )
            await self._connection.commit()
        except Exception:
            pass  # Column already exists

        # Migration: Add eval columns to active_quizzes if they don't exist
        for col in ["eval_before", "eval_after_best", "eval_after_played"]:
            try:
                await self._connection.execute(
                    f"ALTER TABLE active_quizzes ADD COLUMN {col} REAL DEFAULT 0"
                )
                await self._connection.commit()
            except Exception:
                pass  # Column already exists

    # Guild operations
    async def get_or_create_guild(self, guild_id: int) -> Guild:
        """Get or create a guild record."""
        cursor = await self._connection.execute(
            "SELECT * FROM guilds WHERE id = ?", (guild_id,)
        )
        row = await cursor.fetchone()

        if row:
            return Guild(
                id=row["id"],
                notification_channel_id=row["notification_channel_id"],
                summary_channel_id=row["summary_channel_id"],
                summary_time=row["summary_time"],
            )

        await self._connection.execute(
            "INSERT INTO guilds (id) VALUES (?)", (guild_id,)
        )
        await self._connection.commit()
        return Guild(id=guild_id)

    async def update_guild(self, guild: Guild):
        """Update guild settings."""
        await self._connection.execute(
            """UPDATE guilds
               SET notification_channel_id = ?, summary_channel_id = ?, summary_time = ?
               WHERE id = ?""",
            (guild.notification_channel_id, guild.summary_channel_id,
             guild.summary_time, guild.id),
        )
        await self._connection.commit()

    async def get_all_guilds(self) -> list[Guild]:
        """Get all guilds."""
        cursor = await self._connection.execute("SELECT * FROM guilds")
        rows = await cursor.fetchall()
        return [
            Guild(
                id=row["id"],
                notification_channel_id=row["notification_channel_id"],
                summary_channel_id=row["summary_channel_id"],
                summary_time=row["summary_time"],
            )
            for row in rows
        ]

    # Tracked player operations
    async def add_tracked_player(self, player: TrackedPlayer) -> TrackedPlayer:
        """Add a player to track."""
        cursor = await self._connection.execute(
            """INSERT INTO tracked_players
               (guild_id, platform, username, display_name, discord_user_id, added_by, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (player.guild_id, player.platform, player.username.lower(),
             player.display_name, player.discord_user_id, player.added_by, player.added_at),
        )
        await self._connection.commit()
        player.id = cursor.lastrowid
        return player

    async def update_tracked_player(self, player: TrackedPlayer):
        """Update a tracked player's settings."""
        await self._connection.execute(
            """UPDATE tracked_players
               SET display_name = ?, discord_user_id = ?
               WHERE id = ?""",
            (player.display_name, player.discord_user_id, player.id),
        )
        await self._connection.commit()

    async def remove_tracked_player(self, guild_id: int, platform: str, username: str) -> bool:
        """Remove a tracked player."""
        cursor = await self._connection.execute(
            """DELETE FROM tracked_players
               WHERE guild_id = ? AND platform = ? AND username = ?""",
            (guild_id, platform, username.lower()),
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    async def get_tracked_players(self, guild_id: int) -> list[TrackedPlayer]:
        """Get all tracked players for a guild."""
        cursor = await self._connection.execute(
            "SELECT * FROM tracked_players WHERE guild_id = ?", (guild_id,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_player(row) for row in rows]

    async def get_all_tracked_players(self) -> list[TrackedPlayer]:
        """Get all tracked players across all guilds."""
        cursor = await self._connection.execute("SELECT * FROM tracked_players")
        rows = await cursor.fetchall()
        return [self._row_to_player(row) for row in rows]

    async def get_tracked_player_by_id(self, player_id: int) -> Optional[TrackedPlayer]:
        """Get a tracked player by their ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM tracked_players WHERE id = ?", (player_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_player(row) if row else None

    async def get_tracked_player(
        self, guild_id: int, platform: str, username: str
    ) -> Optional[TrackedPlayer]:
        """Get a specific tracked player."""
        cursor = await self._connection.execute(
            """SELECT * FROM tracked_players
               WHERE guild_id = ? AND platform = ? AND username = ?""",
            (guild_id, platform, username.lower()),
        )
        row = await cursor.fetchone()
        return self._row_to_player(row) if row else None

    def _row_to_player(self, row) -> TrackedPlayer:
        """Convert a database row to TrackedPlayer."""
        return TrackedPlayer(
            id=row["id"],
            guild_id=row["guild_id"],
            platform=row["platform"],
            username=row["username"],
            display_name=row["display_name"],
            discord_user_id=row["discord_user_id"] if "discord_user_id" in row.keys() else None,
            added_by=row["added_by"],
            added_at=datetime.fromisoformat(row["added_at"]) if row["added_at"] else None,
        )

    # Game operations
    async def add_game(self, game: Game) -> Optional[Game]:
        """Add a game record. Returns None if game already exists."""
        try:
            cursor = await self._connection.execute(
                """INSERT INTO games
                   (player_id, game_id, platform, time_control, time_control_display,
                    result, player_color, rating_after, rating_change, opponent, opponent_rating,
                    played_at, game_url, final_fen, notified, accuracy, termination)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (game.player_id, game.game_id, game.platform, game.time_control,
                 game.time_control_display, game.result, game.player_color, game.rating_after,
                 game.rating_change, game.opponent, game.opponent_rating,
                 game.played_at, game.game_url, game.final_fen, game.notified, game.accuracy,
                 game.termination),
            )
            await self._connection.commit()
            game.id = cursor.lastrowid
            return game
        except aiosqlite.IntegrityError:
            return None

    async def mark_game_notified(self, game_id: int):
        """Mark a game as notified."""
        await self._connection.execute(
            "UPDATE games SET notified = 1 WHERE id = ?", (game_id,)
        )
        await self._connection.commit()

    async def update_game_accuracy(self, game_id: int, accuracy: float):
        """Update the accuracy for a game."""
        await self._connection.execute(
            "UPDATE games SET accuracy = ? WHERE id = ?", (accuracy, game_id)
        )
        await self._connection.commit()

    async def update_game_termination(self, game_id: int, termination: str):
        """Update the termination for a game."""
        await self._connection.execute(
            "UPDATE games SET termination = ? WHERE id = ?", (termination, game_id)
        )
        await self._connection.commit()

    async def get_games_without_termination(self) -> list[Game]:
        """Get all games that don't have termination set."""
        cursor = await self._connection.execute(
            """SELECT * FROM games
               WHERE termination IS NULL AND final_fen IS NOT NULL
               ORDER BY played_at DESC"""
        )
        rows = await cursor.fetchall()
        return [self._row_to_game(row) for row in rows]

    async def get_games_without_accuracy(self) -> list[Game]:
        """Get all games that don't have accuracy calculated."""
        cursor = await self._connection.execute(
            """SELECT * FROM games
               WHERE accuracy IS NULL
               ORDER BY played_at DESC"""
        )
        rows = await cursor.fetchall()
        return [self._row_to_game(row) for row in rows]

    async def get_unnotified_games(self, player_id: int) -> list[Game]:
        """Get games that haven't been notified yet."""
        cursor = await self._connection.execute(
            """SELECT * FROM games
               WHERE player_id = ? AND notified = 0
               ORDER BY played_at ASC""",
            (player_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_game(row) for row in rows]

    async def get_player_games_since(
        self, player_id: int, since: datetime
    ) -> list[Game]:
        """Get games for a player since a given time."""
        cursor = await self._connection.execute(
            """SELECT * FROM games
               WHERE player_id = ? AND played_at >= ?
               ORDER BY played_at ASC""",
            (player_id, since),
        )
        rows = await cursor.fetchall()
        return [self._row_to_game(row) for row in rows]

    async def get_latest_game_time(self, player_id: int) -> Optional[datetime]:
        """Get the timestamp of the latest game for a player."""
        cursor = await self._connection.execute(
            """SELECT MAX(played_at) as latest FROM games WHERE player_id = ?""",
            (player_id,),
        )
        row = await cursor.fetchone()
        if row and row["latest"]:
            return datetime.fromisoformat(row["latest"])
        return None

    async def get_last_rating(self, player_id: int, time_control: str) -> Optional[int]:
        """Get the rating from the most recent game for a player in a specific time control."""
        cursor = await self._connection.execute(
            """SELECT rating_after FROM games
               WHERE player_id = ? AND time_control = ?
               ORDER BY played_at DESC
               LIMIT 1""",
            (player_id, time_control),
        )
        row = await cursor.fetchone()
        if row and row["rating_after"]:
            return row["rating_after"]
        return None

    async def game_exists(self, player_id: int, game_id: str) -> bool:
        """Check if a game already exists."""
        cursor = await self._connection.execute(
            "SELECT 1 FROM games WHERE player_id = ? AND game_id = ?",
            (player_id, game_id),
        )
        return await cursor.fetchone() is not None

    async def get_player_game_count(self, player_id: int) -> int:
        """Get the total number of games stored for a player."""
        cursor = await self._connection.execute(
            "SELECT COUNT(*) as count FROM games WHERE player_id = ?",
            (player_id,),
        )
        row = await cursor.fetchone()
        return row["count"] if row else 0

    async def get_recent_games(self, player_id: int, limit: int = 5) -> list[Game]:
        """Get the most recent games for a player."""
        cursor = await self._connection.execute(
            """SELECT * FROM games
               WHERE player_id = ?
               ORDER BY played_at DESC
               LIMIT ?""",
            (player_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_game(row) for row in rows]

    async def get_random_lost_game_with_player(
        self, guild_id: int
    ) -> Optional[tuple[TrackedPlayer, Game]]:
        """
        Get a random game where the tracked player lost by checkmate or resignation.

        Args:
            guild_id: The Discord guild ID

        Returns:
            Tuple of (TrackedPlayer, Game) or None if no suitable games found
        """
        cursor = await self._connection.execute(
            """SELECT
                tp.id as tp_id, tp.guild_id, tp.platform as tp_platform,
                tp.username, tp.display_name, tp.discord_user_id,
                tp.added_by, tp.added_at,
                g.id as g_id, g.player_id, g.game_id, g.platform as g_platform,
                g.time_control, g.time_control_display, g.result, g.player_color,
                g.rating_after, g.rating_change, g.opponent, g.opponent_rating,
                g.played_at, g.game_url, g.final_fen, g.notified, g.accuracy, g.termination
               FROM tracked_players tp
               JOIN games g ON g.player_id = tp.id
               WHERE tp.guild_id = ?
                 AND g.result = 'loss'
                 AND g.termination IN ('checkmate', 'resign')
               ORDER BY RANDOM()
               LIMIT 1""",
            (guild_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        player = TrackedPlayer(
            id=row["tp_id"],
            guild_id=row["guild_id"],
            platform=row["tp_platform"],
            username=row["username"],
            display_name=row["display_name"],
            discord_user_id=row["discord_user_id"],
            added_by=row["added_by"],
            added_at=datetime.fromisoformat(row["added_at"]) if row["added_at"] else None,
        )

        game = Game(
            id=row["g_id"],
            player_id=row["player_id"],
            game_id=row["game_id"],
            platform=row["g_platform"],
            time_control=row["time_control"],
            time_control_display=row["time_control_display"],
            result=row["result"],
            player_color=row["player_color"],
            rating_after=row["rating_after"],
            rating_change=row["rating_change"],
            opponent=row["opponent"],
            opponent_rating=row["opponent_rating"],
            played_at=datetime.fromisoformat(row["played_at"]) if row["played_at"] else None,
            game_url=row["game_url"],
            final_fen=row["final_fen"],
            notified=bool(row["notified"]),
            accuracy=row["accuracy"],
            termination=row["termination"],
        )

        return (player, game)

    async def get_random_lost_game_for_player(
        self, player_id: int
    ) -> Optional[tuple[TrackedPlayer, Game]]:
        """
        Get a random game where a specific tracked player lost by checkmate or resignation.

        Args:
            player_id: The tracked player's database ID

        Returns:
            Tuple of (TrackedPlayer, Game) or None if no suitable games found
        """
        cursor = await self._connection.execute(
            """SELECT
                tp.id as tp_id, tp.guild_id, tp.platform as tp_platform,
                tp.username, tp.display_name, tp.discord_user_id,
                tp.added_by, tp.added_at,
                g.id as g_id, g.player_id, g.game_id, g.platform as g_platform,
                g.time_control, g.time_control_display, g.result, g.player_color,
                g.rating_after, g.rating_change, g.opponent, g.opponent_rating,
                g.played_at, g.game_url, g.final_fen, g.notified, g.accuracy, g.termination
               FROM tracked_players tp
               JOIN games g ON g.player_id = tp.id
               WHERE tp.id = ?
                 AND g.result = 'loss'
                 AND g.termination IN ('checkmate', 'resign')
               ORDER BY RANDOM()
               LIMIT 1""",
            (player_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        player = TrackedPlayer(
            id=row["tp_id"],
            guild_id=row["guild_id"],
            platform=row["tp_platform"],
            username=row["username"],
            display_name=row["display_name"],
            discord_user_id=row["discord_user_id"],
            added_by=row["added_by"],
            added_at=datetime.fromisoformat(row["added_at"]) if row["added_at"] else None,
        )

        game = Game(
            id=row["g_id"],
            player_id=row["player_id"],
            game_id=row["game_id"],
            platform=row["g_platform"],
            time_control=row["time_control"],
            time_control_display=row["time_control_display"],
            result=row["result"],
            player_color=row["player_color"],
            rating_after=row["rating_after"],
            rating_change=row["rating_change"],
            opponent=row["opponent"],
            opponent_rating=row["opponent_rating"],
            played_at=datetime.fromisoformat(row["played_at"]) if row["played_at"] else None,
            game_url=row["game_url"],
            final_fen=row["final_fen"],
            notified=bool(row["notified"]),
            accuracy=row["accuracy"],
            termination=row["termination"],
        )

        return (player, game)

    async def get_random_lost_game_global(self) -> Optional[tuple[TrackedPlayer, Game]]:
        """
        Get a random game where any tracked player lost by checkmate or resignation.
        Searches across all guilds/servers.

        Returns:
            Tuple of (TrackedPlayer, Game) or None if no suitable games found
        """
        cursor = await self._connection.execute(
            """SELECT
                tp.id as tp_id, tp.guild_id, tp.platform as tp_platform,
                tp.username, tp.display_name, tp.discord_user_id,
                tp.added_by, tp.added_at,
                g.id as g_id, g.player_id, g.game_id, g.platform as g_platform,
                g.time_control, g.time_control_display, g.result, g.player_color,
                g.rating_after, g.rating_change, g.opponent, g.opponent_rating,
                g.played_at, g.game_url, g.final_fen, g.notified, g.accuracy, g.termination
               FROM tracked_players tp
               JOIN games g ON g.player_id = tp.id
               WHERE g.result = 'loss'
                 AND g.termination IN ('checkmate', 'resign')
               ORDER BY RANDOM()
               LIMIT 1"""
        )
        row = await cursor.fetchone()

        if not row:
            return None

        player = TrackedPlayer(
            id=row["tp_id"],
            guild_id=row["guild_id"],
            platform=row["tp_platform"],
            username=row["username"],
            display_name=row["display_name"],
            discord_user_id=row["discord_user_id"],
            added_by=row["added_by"],
            added_at=datetime.fromisoformat(row["added_at"]) if row["added_at"] else None,
        )

        game = Game(
            id=row["g_id"],
            player_id=row["player_id"],
            game_id=row["game_id"],
            platform=row["g_platform"],
            time_control=row["time_control"],
            time_control_display=row["time_control_display"],
            result=row["result"],
            player_color=row["player_color"],
            rating_after=row["rating_after"],
            rating_change=row["rating_change"],
            opponent=row["opponent"],
            opponent_rating=row["opponent_rating"],
            played_at=datetime.fromisoformat(row["played_at"]) if row["played_at"] else None,
            game_url=row["game_url"],
            final_fen=row["final_fen"],
            notified=bool(row["notified"]),
            accuracy=row["accuracy"],
            termination=row["termination"],
        )

        return (player, game)

    async def get_game_by_id(self, player_id: int, game_id: str) -> Optional[Game]:
        """Get a specific game by its platform game ID."""
        cursor = await self._connection.execute(
            """SELECT * FROM games WHERE player_id = ? AND game_id = ?""",
            (player_id, game_id),
        )
        row = await cursor.fetchone()
        return self._row_to_game(row) if row else None

    async def get_stats_last_24h(self, player_id: int) -> dict:
        """Get statistics for a player from the last 24 hours."""
        now = datetime.utcnow()
        since = now - timedelta(hours=24)
        return await self._get_stats_since(player_id, since)

    async def get_daily_stats(
        self, player_id: int, date: Optional[datetime] = None
    ) -> dict:
        """Get daily statistics for a player."""
        if date is None:
            date = datetime.utcnow()

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        cursor = await self._connection.execute(
            """SELECT
                time_control,
                result,
                rating_after,
                rating_change
               FROM games
               WHERE player_id = ? AND played_at >= ? AND played_at < ?
               ORDER BY played_at ASC""",
            (player_id, start, end),
        )
        rows = await cursor.fetchall()

        stats = {
            "total_games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "by_time_control": {},
        }

        for row in rows:
            stats["total_games"] += 1
            if row["result"] == "win":
                stats["wins"] += 1
            elif row["result"] == "loss":
                stats["losses"] += 1
            else:
                stats["draws"] += 1

            tc = row["time_control"]
            if tc not in stats["by_time_control"]:
                stats["by_time_control"][tc] = {
                    "games": 0,
                    "rating_change": 0,
                    "final_rating": 0,
                }

            stats["by_time_control"][tc]["games"] += 1
            stats["by_time_control"][tc]["rating_change"] += row["rating_change"] or 0
            stats["by_time_control"][tc]["final_rating"] = row["rating_after"] or 0

        return stats

    async def _get_stats_since(self, player_id: int, since: datetime) -> dict:
        """Get statistics for a player since a given time."""
        cursor = await self._connection.execute(
            """SELECT
                time_control,
                result,
                rating_after,
                rating_change
               FROM games
               WHERE player_id = ? AND played_at >= ?
               ORDER BY played_at ASC""",
            (player_id, since),
        )
        rows = await cursor.fetchall()

        stats = {
            "total_games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "by_time_control": {},
        }

        for row in rows:
            stats["total_games"] += 1
            if row["result"] == "win":
                stats["wins"] += 1
            elif row["result"] == "loss":
                stats["losses"] += 1
            else:
                stats["draws"] += 1

            tc = row["time_control"]
            if tc not in stats["by_time_control"]:
                stats["by_time_control"][tc] = {
                    "games": 0,
                    "rating_change": 0,
                    "final_rating": 0,
                }

            stats["by_time_control"][tc]["games"] += 1
            stats["by_time_control"][tc]["rating_change"] += row["rating_change"] or 0
            stats["by_time_control"][tc]["final_rating"] = row["rating_after"] or 0

        return stats

    def _row_to_game(self, row) -> Game:
        """Convert a database row to Game."""
        return Game(
            id=row["id"],
            player_id=row["player_id"],
            game_id=row["game_id"],
            platform=row["platform"],
            time_control=row["time_control"],
            time_control_display=row["time_control_display"],
            result=row["result"],
            player_color=row["player_color"],
            rating_after=row["rating_after"],
            rating_change=row["rating_change"],
            opponent=row["opponent"],
            opponent_rating=row["opponent_rating"],
            played_at=datetime.fromisoformat(row["played_at"]) if row["played_at"] else None,
            game_url=row["game_url"],
            final_fen=row["final_fen"],
            notified=bool(row["notified"]),
            accuracy=row["accuracy"] if "accuracy" in row.keys() else None,
            termination=row["termination"] if "termination" in row.keys() else None,
        )

    # Quiz operations
    async def save_quiz(self, quiz: ActiveQuiz) -> ActiveQuiz:
        """Save an active quiz to the database."""
        await self._connection.execute(
            """INSERT OR REPLACE INTO active_quizzes
               (channel_id, guild_id, position_fen, correct_move_san, played_move_san,
                game_url, player_username, opponent_username, move_number, difficulty,
                eval_before, eval_after_best, eval_after_played, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (quiz.channel_id, quiz.guild_id, quiz.position_fen, quiz.correct_move_san,
             quiz.played_move_san, quiz.game_url, quiz.player_username, quiz.opponent_username,
             quiz.move_number, quiz.difficulty, quiz.eval_before, quiz.eval_after_best,
             quiz.eval_after_played, quiz.started_at),
        )
        await self._connection.commit()
        return quiz

    async def get_quiz(self, channel_id: int) -> Optional[ActiveQuiz]:
        """Get an active quiz by channel ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM active_quizzes WHERE channel_id = ?",
            (channel_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_quiz(row) if row else None

    async def delete_quiz(self, channel_id: int) -> bool:
        """Delete an active quiz."""
        cursor = await self._connection.execute(
            "DELETE FROM active_quizzes WHERE channel_id = ?",
            (channel_id,),
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    def _row_to_quiz(self, row) -> ActiveQuiz:
        """Convert a database row to ActiveQuiz."""
        return ActiveQuiz(
            channel_id=row["channel_id"],
            guild_id=row["guild_id"],
            position_fen=row["position_fen"],
            correct_move_san=row["correct_move_san"],
            played_move_san=row["played_move_san"],
            game_url=row["game_url"],
            player_username=row["player_username"],
            opponent_username=row["opponent_username"],
            move_number=row["move_number"],
            difficulty=row["difficulty"],
            eval_before=row["eval_before"],
            eval_after_best=row["eval_after_best"],
            eval_after_played=row["eval_after_played"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        )

    # Quiz score operations
    async def add_quiz_point(self, guild_id: int, user_id: int, username: str) -> int:
        """
        Add a point to a user's quiz score.

        Args:
            guild_id: The Discord guild ID
            user_id: The Discord user ID
            username: The user's display name (updated on each point)

        Returns:
            The user's new total score
        """
        # Use INSERT OR REPLACE with score increment
        await self._connection.execute(
            """INSERT INTO quiz_scores (guild_id, user_id, username, score)
               VALUES (?, ?, ?, 1)
               ON CONFLICT(guild_id, user_id) DO UPDATE SET
                   score = score + 1,
                   username = excluded.username""",
            (guild_id, user_id, username),
        )
        await self._connection.commit()

        # Get the new score
        cursor = await self._connection.execute(
            "SELECT score FROM quiz_scores WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        )
        row = await cursor.fetchone()
        return row["score"] if row else 1

    async def get_quiz_leaderboard(
        self, guild_id: int, limit: int = 10
    ) -> list[tuple[int, str, int]]:
        """
        Get the quiz leaderboard for a guild.

        Args:
            guild_id: The Discord guild ID
            limit: Maximum number of entries to return

        Returns:
            List of (user_id, username, score) tuples, ordered by score descending
        """
        cursor = await self._connection.execute(
            """SELECT user_id, username, score FROM quiz_scores
               WHERE guild_id = ?
               ORDER BY score DESC
               LIMIT ?""",
            (guild_id, limit),
        )
        rows = await cursor.fetchall()
        return [(row["user_id"], row["username"], row["score"]) for row in rows]

    async def get_user_quiz_score(self, guild_id: int, user_id: int) -> int:
        """Get a user's quiz score in a guild."""
        cursor = await self._connection.execute(
            "SELECT score FROM quiz_scores WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        )
        row = await cursor.fetchone()
        return row["score"] if row else 0
