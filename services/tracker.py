import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from api import ChessComClient, LichessClient
from api.base import GameData
from config import POLL_INTERVAL, Platform
from database import DatabaseManager, Game, TrackedPlayer
from utils.accuracy import calculate_accuracy_from_pgn

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
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}", exc_info=True)

            await asyncio.sleep(POLL_INTERVAL)

    async def _poll_all_players(self):
        """Poll all tracked players for new games."""
        players = await self.db.get_all_tracked_players()

        if not players:
            logger.debug("No tracked players found")
            return

        logger.info(f"Polling {len(players)} tracked player(s)")

        # Group by platform to avoid mixing rate limits
        chesscom_players = [p for p in players if p.platform == Platform.CHESSCOM]
        lichess_players = [p for p in players if p.platform == Platform.LICHESS]

        # Poll Chess.com players in parallel
        if chesscom_players:
            logger.debug(f"Polling {len(chesscom_players)} Chess.com player(s)")
            tasks = [self._poll_player(p, self.chesscom_client) for p in chesscom_players]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for player, result in zip(chesscom_players, results):
                if isinstance(result, Exception):
                    logger.error(f"Error polling {player.username} on Chess.com: {result}")

        # Poll Lichess players in parallel
        if lichess_players:
            logger.debug(f"Polling {len(lichess_players)} Lichess player(s)")
            tasks = [self._poll_player(p, self.lichess_client) for p in lichess_players]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for player, result in zip(lichess_players, results):
                if isinstance(result, Exception):
                    logger.error(f"Error polling {player.username} on Lichess: {result}")

    async def _poll_player(self, player: TrackedPlayer, client):
        """Poll a single player for new games."""
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

        # Calculate accuracy from PGN
        accuracy = None
        if pgn:
            try:
                accuracy = await calculate_accuracy_from_pgn(pgn, game_data.player_color)
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
        )

        # Store in database
        saved_game = await self.db.add_game(game)
        if saved_game:
            logger.info(f"New game recorded: {player.username} vs {game_data.opponent}")

            # Send notification with video
            await self.notification_service.send_game_notification(player, saved_game, pgn)

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

    async def initialize_player_history(self, player: TrackedPlayer) -> int:
        """Fetch and store game history for a newly tracked player.

        Fetches games from the last 24 hours for summary purposes. If no games
        are found in that window, fetches the most recent game from history
        so future polling has a reference point.

        Games are stored silently without sending notifications.
        """
        if player.platform == Platform.CHESSCOM:
            client = self.chesscom_client
        else:
            client = self.lichess_client

        try:
            # Fetch games from the last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            games = await client.get_recent_games(player.username, since=since)

            # If no games in last 24 hours, fetch the most recent game from history
            # so we have a reference point for future polling
            if not games:
                logger.info(f"No games in last 24h for {player.username}, fetching most recent game")
                historical_games = await client.get_games_for_analysis(player.username, max_games=1)
                games = historical_games[:1] if historical_games else []

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

    async def refresh_guild(self, guild_id: int) -> int:
        """Manually refresh all tracked players for a guild. Returns number of new games found."""
        players = await self.db.get_tracked_players(guild_id)
        new_games = 0

        for player in players:
            if player.platform == Platform.CHESSCOM:
                client = self.chesscom_client
            else:
                client = self.lichess_client

            try:
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

        return new_games
