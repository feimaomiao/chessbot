import asyncio
import hashlib
import io
import logging
import time
from typing import Optional

import discord

from config import DISABLE_VIDEO
from database import DatabaseManager, Game, TrackedPlayer
from utils.board import get_board_discord_file
from utils.helpers import (
    format_rating_change,
    get_result_emoji,
    get_time_control_emoji,
)
from utils.video import generate_game_video_async

logger = logging.getLogger(__name__)

# Video cache for multi-server optimization
# Key: PGN hash, Value: (video_bytes, timestamp)
# Expires after 5 minutes (enough time for all guild notifications to complete)
_video_cache: dict[str, tuple[bytes, float]] = {}
_VIDEO_CACHE_TTL = 300  # 5 minutes
_video_cache_locks: dict[str, asyncio.Lock] = {}
_video_cache_master_lock = asyncio.Lock()


class NotificationService:
    """Service for sending Discord notifications."""

    def __init__(self, bot: discord.Client, db: DatabaseManager):
        self.bot = bot
        self.db = db

    async def send_game_notification(
        self, player: TrackedPlayer, game: Game, pgn: Optional[str] = None,
        evaluations: Optional[list[float]] = None
    ):
        """Send a notification for a completed game."""
        # Get guild settings
        guild = await self.db.get_or_create_guild(player.guild_id)

        if not guild.notification_channel_id:
            logger.debug(f"No notification channel set for guild {player.guild_id}")
            return

        # Try cache first, then fetch
        channel = self.bot.get_channel(guild.notification_channel_id)
        if not channel:
            try:
                channel = await self.bot.fetch_channel(guild.notification_channel_id)
            except discord.NotFound:
                logger.warning(f"Channel {guild.notification_channel_id} not found")
                return
            except discord.Forbidden:
                logger.warning(f"No access to channel {guild.notification_channel_id}")
                return

        # Create embed and file (video if PGN available, otherwise static image)
        embed, file = await self.create_game_message(player, game, pgn, evaluations)

        # Ping linked Discord member if set
        content = None
        if player.discord_user_id:
            content = f"<@{player.discord_user_id}>"

        try:
            await channel.send(content=content, embed=embed, file=file)
            logger.info(f"Sent notification for {player.username}'s game")
        except discord.Forbidden:
            logger.error(f"No permission to send to channel {channel.id}")
        except discord.HTTPException as e:
            logger.error(f"Failed to send notification: {e}")

    async def create_game_message(
        self, player: TrackedPlayer, game: Game, pgn: Optional[str] = None,
        evaluations: Optional[list[float]] = None
    ) -> tuple[discord.Embed, discord.File | None]:
        """Create an embed and video/image for a game notification."""
        display_name = player.display_name or player.username

        # Determine white and black players
        if game.player_color == "white":
            white_name = display_name
            white_rating = game.rating_after
            black_name = game.opponent
            black_rating = game.opponent_rating
        else:
            white_name = game.opponent
            white_rating = game.opponent_rating
            black_name = display_name
            black_rating = game.rating_after

        # Title format: White (Rating) vs Black (Rating)
        title = f"{white_name} ({white_rating}) vs {black_name} ({black_rating})"

        # Result emoji
        result_emoji = get_result_emoji(game.result)

        # Color based on result
        if game.result == "win":
            embed_color = discord.Color.green()
        elif game.result == "loss":
            embed_color = discord.Color.red()
        else:
            embed_color = discord.Color.greyple()

        # Time control formatting
        tc_emoji = get_time_control_emoji(game.time_control)

        # Rating change formatting
        rating_change_str = format_rating_change(game.rating_change)

        # Timestamp
        time_str = game.played_at.strftime("%b %d, %Y at %H:%M UTC") if game.played_at else "Unknown"

        # Create embed
        embed = discord.Embed(
            title=title,
            color=embed_color,
            url=game.game_url,
        )

        # Format termination for display
        termination_display = None
        if game.termination and game.termination != "unknown":
            termination_map = {
                "checkmate": "Checkmate",
                "timeout": "Timeout",
                "resign": "Resignation",
                "aborted": "Aborted",
                "agreed": "Draw Agreement",
                "stalemate": "Stalemate",
                "repetition": "Repetition",
            }
            termination_display = termination_map.get(game.termination, game.termination.capitalize())

        # Game info in description
        result_text = f"{result_emoji} **{game.result.capitalize()}**"
        if termination_display:
            result_text += f" by {termination_display}"

        description_lines = [
            result_text,
            f"**Format:** {tc_emoji} {game.time_control.capitalize()} ({game.time_control_display})",
            f"**Rating:** {game.rating_after} ({rating_change_str})",
        ]

        # Add accuracy if available
        if game.accuracy is not None:
            description_lines.append(f"**Accuracy:** {game.accuracy:.1f}%")

        description_lines.append(f"**Played:** {time_str}")
        embed.description = "\n".join(description_lines)

        # Generate video if PGN is available and video is enabled
        file = None
        if pgn and not DISABLE_VIDEO:
            try:
                # Check video cache first (for multi-server optimization)
                pgn_hash = hashlib.md5(pgn.encode()).hexdigest()

                # Get or create a lock for this specific PGN
                async with _video_cache_master_lock:
                    if pgn_hash not in _video_cache_locks:
                        _video_cache_locks[pgn_hash] = asyncio.Lock()
                    pgn_lock = _video_cache_locks[pgn_hash]

                # Acquire lock for this PGN - only one task generates, others wait
                async with pgn_lock:
                    now = time.time()

                    # Clean expired entries
                    expired = [k for k, (_, ts) in _video_cache.items() if now - ts > _VIDEO_CACHE_TTL]
                    for k in expired:
                        del _video_cache[k]
                        _video_cache_locks.pop(k, None)

                    if pgn_hash in _video_cache:
                        video_bytes, _ = _video_cache[pgn_hash]
                        logger.info(f"Using cached video for {player.username}'s game")
                    else:
                        logger.info(f"Generating video for {player.username}'s game...")
                        # Pass pre-computed evaluations to avoid re-evaluation
                        # Render from tracked player's perspective
                        video_bytes = await generate_game_video_async(
                            pgn, evaluations=evaluations, player_color=game.player_color
                        )
                        if video_bytes:
                            _video_cache[pgn_hash] = (video_bytes, now)
                            logger.info(f"Video generated and cached for {player.username}'s game")

                if video_bytes:
                    file = discord.File(
                        io.BytesIO(video_bytes),
                        filename=f"game_{game.game_id}.mp4",
                    )
            except Exception as e:
                logger.error(f"Failed to generate video: {e}")

        # Fall back to static board image if video failed or no PGN
        if file is None and game.final_fen:
            # Render from tracked player's perspective
            flipped = game.player_color == "black"
            file = get_board_discord_file(game.final_fen, f"board_{game.game_id}.png", flipped=flipped)
            if file:
                embed.set_image(url=f"attachment://board_{game.game_id}.png")

        return embed, file

    async def send_message(self, channel_id: int, content: str = None, embed: discord.Embed = None):
        """Send a message to a channel."""
        channel = self.bot.get_channel(channel_id)
        if not channel:
            logger.warning(f"Could not find channel {channel_id}")
            return

        try:
            await channel.send(content=content, embed=embed)
        except discord.Forbidden:
            logger.error(f"No permission to send to channel {channel_id}")
        except discord.HTTPException as e:
            logger.error(f"Failed to send message: {e}")
