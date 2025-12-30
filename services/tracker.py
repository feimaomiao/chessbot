import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from api import ChessComClient, LichessClient
from api.base import GameData
from config import POLL_INTERVAL, Platform
from database import DatabaseManager, Game, TrackedPlayer
from utils.accuracy import evaluate_game_positions, calculate_accuracy_from_evaluations
from utils.video import get_eval_cache

if TYPE_CHECKING:
    from services.notifications import NotificationService

logger = logging.getLogger(__name__)


class GameTracker:
    """Service for tracking chess games across platforms."""

    def __init__(self, db: DatabaseManager, notification_service: "NotificationService"):
        self.db = db
        self.notification_service = notification_service
        self.chesscom_client = ChessComClient()
        self.lichess_client = LichessClient()
        self._running = False
        self._task: asyncio.Task = None

    async def start(self):
        """Start the tracking loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._tracking_loop())
        logger.info("Game tracker started")

    async def stop(self):
        """Stop the tracking loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.chesscom_client.close()
        await self.lichess_client.close()
        logger.info("Game tracker stopped")

    async def _tracking_loop(self):
        """Main tracking loop."""
        while self._running:
            try:
                logger.debug("Starting poll cycle...")
                await self._poll_all_players()
                logger.debug("Poll cycle complete")
                # Save evaluation cache if modified
                get_eval_cache().save_if_dirty()
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}", exc_info=True)

            await asyncio.sleep(POLL_INTERVAL)

    async def _poll_all_players(self):
        """Poll all tracked players for new games."""
        players = await self.db.get_all_tracked_players()

        if not players:
            logger.debug("No tracked players found")
            return

        # Group players by (platform, username) to avoid duplicate API calls
        # when the same user is tracked in multiple servers
        player_groups: dict[tuple[str, str], list[TrackedPlayer]] = {}
        for p in players:
            key = (p.platform, p.username.lower())
            if key not in player_groups:
                player_groups[key] = []
            player_groups[key].append(p)

        unique_users = len(player_groups)
        total_trackers = len(players)
        if unique_users < total_trackers:
            logger.info(f"Polling {unique_users} unique user(s) ({total_trackers} trackers)")
        else:
            logger.info(f"Polling {unique_users} user(s)")

        # Group by platform to avoid mixing rate limits
        chesscom_groups = {k: v for k, v in player_groups.items() if k[0] == Platform.CHESSCOM}
        lichess_groups = {k: v for k, v in player_groups.items() if k[0] == Platform.LICHESS}

        total_new_games = 0

        # Poll Chess.com users in parallel (one API call per unique user)
        if chesscom_groups:
            logger.debug(f"Polling {len(chesscom_groups)} unique Chess.com user(s)")
            tasks = [
                self._poll_user_group(players_list, self.chesscom_client)
                for players_list in chesscom_groups.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (platform, username), result in zip(chesscom_groups.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Error polling {username} on Chess.com: {result}")
                elif isinstance(result, int):
                    total_new_games += result

        # Poll Lichess users in parallel (one API call per unique user)
        if lichess_groups:
            logger.debug(f"Polling {len(lichess_groups)} unique Lichess user(s)")
            tasks = [
                self._poll_user_group(players_list, self.lichess_client)
                for players_list in lichess_groups.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (platform, username), result in zip(lichess_groups.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Error polling {username} on Lichess: {result}")
                elif isinstance(result, int):
                    total_new_games += result

        if total_new_games == 0:
            logger.info("No new games found")

    async def _poll_user_group(self, players: list[TrackedPlayer], client) -> int:
        """Poll a unique user and process games for all their trackers.

        Returns:
            Number of new games found.
        """
        # Use first player for API call (all have same username/platform)
        representative = players[0]
        username = representative.username
        platform = representative.platform

        logger.debug(f"Polling {username} on {platform}...")

        try:
            # Get the earliest last_game_time across all trackers for this user
            last_game_times = []
            for player in players:
                t = await self.db.get_latest_game_time(player.id)
                if t:
                    last_game_times.append(t)

            if last_game_times:
                last_game_time = min(last_game_times)
            else:
                last_game_time = datetime.utcnow() - timedelta(hours=1)

            # Single API call for this user
            games = await client.get_recent_games(username, since=last_game_time)

            if games:
                logger.info(f"Found {len(games)} new game(s) for {username} on {platform}")
            else:
                logger.debug(f"No new games for {username}")

            # Process each game for all trackers of this user
            for game_data in games:
                for player in players:
                    await self._process_game(player, game_data)

            return len(games)

        except Exception as e:
            logger.error(f"Error polling {username} on {platform}: {e}", exc_info=True)
            raise

    async def _poll_player(self, player: TrackedPlayer, client):
        """Poll a single player for new games. (Legacy method for single-server use)"""
        logger.debug(f"Polling {player.username} on {player.platform}...")

        try:
            # Get the timestamp of their last known game
            last_game_time = await self.db.get_latest_game_time(player.id)

            # If no games recorded, only look back 1 hour to avoid spam
            if last_game_time is None:
                last_game_time = datetime.utcnow() - timedelta(hours=1)

            # Fetch recent games from API
            games = await client.get_recent_games(player.username, since=last_game_time)

            if games:
                logger.info(f"Found {len(games)} new game(s) for {player.username} on {player.platform}")
            else:
                logger.debug(f"No new games for {player.username}")

            for game_data in games:
                await self._process_game(player, game_data)

        except Exception as e:
            logger.error(f"Error polling {player.username} on {player.platform}: {e}", exc_info=True)
            raise  # Re-raise so asyncio.gather can catch it

    async def _process_game(self, player: TrackedPlayer, game_data: GameData):
        """Process a single game from the API."""
        # Check if game already exists
        if await self.db.game_exists(player.id, game_data.game_id):
            return

        # Calculate rating change from previous game if not provided by API
        # Each time control category has its own rating
        rating_change = game_data.rating_change
        if rating_change == 0 and game_data.rating_after:
            last_rating = await self.db.get_last_rating(player.id, game_data.time_control)
            if last_rating:
                rating_change = game_data.rating_after - last_rating

        # Fetch PGN for accuracy calculation and video generation
        pgn = await self._fetch_game_pgn(player, game_data.game_id)

        # Evaluate positions once (used for both accuracy and video)
        evaluations = None
        accuracy = None
        if pgn:
            try:
                evaluations = await evaluate_game_positions(pgn)
                if evaluations:
                    accuracy = calculate_accuracy_from_evaluations(evaluations, game_data.player_color)
                    if accuracy:
                        logger.info(f"Calculated accuracy for {player.username}: {accuracy}%")
            except Exception as e:
                logger.error(f"Error calculating accuracy: {e}")

        # Create game record
        game = Game(
            id=None,
            player_id=player.id,
            game_id=game_data.game_id,
            platform=game_data.platform,
            time_control=game_data.time_control,
            time_control_display=game_data.time_control_display,
            result=game_data.result,
            player_color=game_data.player_color,
            rating_after=game_data.rating_after,
            rating_change=rating_change,
            opponent=game_data.opponent,
            opponent_rating=game_data.opponent_rating,
            played_at=game_data.played_at,
            game_url=game_data.game_url,
            final_fen=game_data.final_fen,
            notified=False,
            accuracy=accuracy,
            termination=game_data.termination,
        )

        # Store in database
        saved_game = await self.db.add_game(game)
        if saved_game:
            logger.info(f"New game recorded: {player.username} vs {game_data.opponent}")

            # Send notification with video (pass evaluations to avoid re-evaluation)
            await self.notification_service.send_game_notification(player, saved_game, pgn, evaluations)

            # Mark as notified
            await self.db.mark_game_notified(saved_game.id)

    async def _fetch_game_pgn(self, player: TrackedPlayer, game_id: str) -> str | None:
        """Fetch PGN for a game from the appropriate API."""
        try:
            if player.platform == Platform.CHESSCOM:
                return await self.chesscom_client.get_game_pgn(player.username, game_id)
            elif player.platform == Platform.LICHESS:
                return await self.lichess_client.get_game_pgn(game_id)
        except Exception as e:
            logger.error(f"Failed to fetch PGN for game {game_id}: {e}")
        return None

    async def validate_player(self, platform: str, username: str) -> bool:
        """Validate that a player exists on the given platform."""
        if platform == Platform.CHESSCOM:
            return await self.chesscom_client.validate_player(username)
        elif platform == Platform.LICHESS:
            return await self.lichess_client.validate_player(username)
        return False

    async def initialize_player_history(
        self, player: TrackedPlayer, max_games: int = 1000
    ) -> int:
        """Fetch and store game history for a newly tracked player.

        Backfills up to max_games from the player's history to populate
        the database for quiz functionality. Games are stored silently
        without sending notifications or calculating accuracy.

        Args:
            player: The tracked player to initialize
            max_games: Maximum number of games to backfill (default: 1000)
        """
        if player.platform == Platform.CHESSCOM:
            client = self.chesscom_client
        else:
            client = self.lichess_client

        try:
            # Backfill historical games for quiz availability
            logger.info(f"Backfilling up to {max_games} games for {player.username}...")
            games = await client.get_games_for_analysis(player.username, max_games=max_games)

            if not games:
                logger.info(f"No games found for {player.username}")
                return 0

            logger.info(f"Fetched {len(games)} games for {player.username}")

            # Sort oldest first for proper rating calculation
            games = sorted(games, key=lambda g: g.played_at)

            stored = 0
            # Track previous rating per time control category
            prev_ratings: dict[str, int] = {}

            for game_data in games:
                # Calculate rating change from previous game in same time control
                rating_change = game_data.rating_change
                tc = game_data.time_control
                if rating_change == 0 and game_data.rating_after and tc in prev_ratings:
                    rating_change = game_data.rating_after - prev_ratings[tc]

                game = Game(
                    id=None,
                    player_id=player.id,
                    game_id=game_data.game_id,
                    platform=game_data.platform,
                    time_control=game_data.time_control,
                    time_control_display=game_data.time_control_display,
                    result=game_data.result,
                    player_color=game_data.player_color,
                    rating_after=game_data.rating_after,
                    rating_change=rating_change,
                    opponent=game_data.opponent,
                    opponent_rating=game_data.opponent_rating,
                    played_at=game_data.played_at,
                    game_url=game_data.game_url,
                    final_fen=game_data.final_fen,
                    notified=True,  # Mark as notified to avoid sending notifications
                    termination=game_data.termination,
                )

                saved = await self.db.add_game(game)
                if saved:
                    stored += 1
                    # Update previous rating for this time control
                    if game_data.rating_after:
                        prev_ratings[tc] = game_data.rating_after

            logger.info(f"Initialized {stored} games for {player.username}")
            return stored

        except Exception as e:
            logger.error(f"Error initializing history for {player.username}: {e}")
            return 0

    async def refresh_guild(self, guild_id: int, min_games: int = 1000) -> tuple[int, int]:
        """Manually refresh all tracked players for a guild.

        Also backfills historical games for players with fewer than min_games.

        Returns:
            Tuple of (new_games_count, backfilled_games_count)
        """
        players = await self.db.get_tracked_players(guild_id)
        new_games = 0
        backfilled_games = 0

        for player in players:
            if player.platform == Platform.CHESSCOM:
                client = self.chesscom_client
            else:
                client = self.lichess_client

            try:
                # Check if player needs backfill
                game_count = await self.db.get_player_game_count(player.id)
                if game_count < min_games:
                    logger.info(
                        f"{player.username} has {game_count} games, backfilling to {min_games}..."
                    )
                    backfilled = await self.initialize_player_history(
                        player, max_games=min_games
                    )
                    # Subtract existing games since initialize_player_history skips duplicates
                    backfilled_games += max(0, backfilled - game_count)

                # Also fetch any new games since last poll
                last_game_time = await self.db.get_latest_game_time(player.id)
                if last_game_time is None:
                    last_game_time = datetime.utcnow() - timedelta(hours=1)

                games = await client.get_recent_games(player.username, since=last_game_time)

                for game_data in games:
                    if not await self.db.game_exists(player.id, game_data.game_id):
                        await self._process_game(player, game_data)
                        new_games += 1

            except Exception as e:
                logger.error(f"Error refreshing {player.username}: {e}")

        return new_games, backfilled_games
