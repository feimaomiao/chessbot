import io
import logging
from typing import Optional

import discord

from database import DatabaseManager, Game, TrackedPlayer
from utils.board import get_board_discord_file
from utils.helpers import (
    format_rating_change,
    get_result_emoji,
    get_time_control_emoji,
)
from utils.video import generate_game_video_async

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending Discord notifications."""

    def __init__(self, bot: discord.Client, db: DatabaseManager):
        self.bot = bot
        self.db = db

    async def send_game_notification(
        self, player: TrackedPlayer, game: Game, pgn: Optional[str] = None
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
        embed, file = await self.create_game_message(player, game, pgn)

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
        self, player: TrackedPlayer, game: Game, pgn: Optional[str] = None
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

        # Game info in description
        embed.description = (
            f"{result_emoji} **{game.result.capitalize()}**\n"
            f"**Format:** {tc_emoji} {game.time_control.capitalize()} ({game.time_control_display})\n"
            f"**Rating:** {game.rating_after} ({rating_change_str})\n"
            f"**Played:** {time_str}"
        )

        # Generate video if PGN is available, otherwise fall back to static image
        file = None
        if pgn:
            try:
                logger.info(f"Generating video for {player.username}'s game...")
                video_bytes = await generate_game_video_async(pgn)
                if video_bytes:
                    file = discord.File(
                        io.BytesIO(video_bytes),
                        filename=f"game_{game.game_id}.mp4",
                    )
                    logger.info(f"Video generated for {player.username}'s game")
            except Exception as e:
                logger.error(f"Failed to generate video: {e}")

        # Fall back to static board image if video failed or no PGN
        if file is None and game.final_fen:
            file = get_board_discord_file(game.final_fen, f"board_{game.game_id}.png")
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
