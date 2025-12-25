import aiosqlite
from datetime import datetime, timedelta
from typing import Optional
from .models import Guild, TrackedPlayer, Game


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
                FOREIGN KEY (player_id) REFERENCES tracked_players(id),
                UNIQUE(player_id, game_id)
            );

            CREATE INDEX IF NOT EXISTS idx_games_player_id ON games(player_id);
            CREATE INDEX IF NOT EXISTS idx_games_played_at ON games(played_at);
            CREATE INDEX IF NOT EXISTS idx_tracked_players_guild ON tracked_players(guild_id);
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
                    played_at, game_url, final_fen, notified, accuracy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (game.player_id, game.game_id, game.platform, game.time_control,
                 game.time_control_display, game.result, game.player_color, game.rating_after,
                 game.rating_change, game.opponent, game.opponent_rating,
                 game.played_at, game.game_url, game.final_fen, game.notified, game.accuracy),
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
        )
