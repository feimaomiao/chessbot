import io
import logging
from typing import Literal

import discord
from discord import app_commands
from discord.ext import commands

from config import Platform
from database import DatabaseManager, TrackedPlayer
from services.tracker import GameTracker
from utils.board import get_board_discord_file
from utils.helpers import format_platform_name, format_rating_change, get_time_control_emoji
from utils.video import generate_game_video_async

# format_platform_name still used for /list and header messages

logger = logging.getLogger(__name__)

PlatformChoice = Literal["chesscom", "lichess"]


class TrackingCog(commands.Cog):
    """Commands for tracking chess players."""

    def __init__(self, bot: commands.Bot, db: DatabaseManager, tracker: GameTracker):
        self.bot = bot
        self.db = db
        self.tracker = tracker

    @app_commands.command(name="track", description="Start tracking a chess player")
    @app_commands.describe(
        platform="The chess platform (chesscom or lichess)",
        username="The player's username on the platform",
        member="Optional Discord member to ping when they play games",
        display_name="Optional custom display name for notifications",
    )
    async def track(
        self,
        interaction: discord.Interaction,
        platform: PlatformChoice,
        username: str,
        member: discord.Member = None,
        display_name: str = None,
    ):
        """Add a player to the tracking list."""
        await interaction.response.defer(ephemeral=True)

        # Validate the player exists
        exists = await self.tracker.validate_player(platform, username)
        if not exists:
            platform_name = format_platform_name(platform)
            await interaction.followup.send(
                f"Player `{username}` not found on {platform_name}. "
                "Please check the username and try again.",
                ephemeral=True,
            )
            return

        # Check if already tracking
        existing = await self.db.get_tracked_player(
            interaction.guild_id, platform, username
        )
        if existing:
            await interaction.followup.send(
                f"Already tracking `{username}` on {format_platform_name(platform)}.",
                ephemeral=True,
            )
            return

        # Ensure guild exists in database
        await self.db.get_or_create_guild(interaction.guild_id)

        # Add player
        player = TrackedPlayer(
            id=None,
            guild_id=interaction.guild_id,
            platform=platform,
            username=username,
            display_name=display_name,
            discord_user_id=member.id if member else None,
            added_by=interaction.user.id,
        )

        player = await self.db.add_tracked_player(player)

        # Fetch initial game history from last 24 hours
        games_added = await self.tracker.initialize_player_history(player)

        display = display_name or username
        platform_name = format_platform_name(platform)

        msg = f"Now tracking **{display}** ({username}) on {platform_name}!"
        if member:
            msg += f"\nLinked to {member.mention}"
        if games_added > 0:
            msg += f"\nLoaded {games_added} game(s) from the last 24 hours."

        await interaction.followup.send(msg, ephemeral=True)

    @app_commands.command(name="untrack", description="Stop tracking a chess player")
    @app_commands.describe(
        platform="The chess platform",
        username="The player's username to stop tracking",
    )
    async def untrack(
        self,
        interaction: discord.Interaction,
        platform: PlatformChoice,
        username: str,
    ):
        """Remove a player from the tracking list."""
        removed = await self.db.remove_tracked_player(
            interaction.guild_id, platform, username
        )

        if removed:
            await interaction.response.send_message(
                f"Stopped tracking `{username}` on {format_platform_name(platform)}.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"`{username}` was not being tracked on {format_platform_name(platform)}.",
                ephemeral=True,
            )

    @app_commands.command(name="link", description="Link a tracked player to a Discord member")
    @app_commands.describe(
        platform="The chess platform",
        username="The player's username",
        member="The Discord member to link (leave empty to unlink)",
    )
    async def link(
        self,
        interaction: discord.Interaction,
        platform: PlatformChoice,
        username: str,
        member: discord.Member = None,
    ):
        """Link or unlink a tracked player to a Discord member."""
        player = await self.db.get_tracked_player(
            interaction.guild_id, platform, username
        )

        if not player:
            await interaction.response.send_message(
                f"`{username}` is not being tracked on {format_platform_name(platform)}. "
                "Use `/track` first.",
                ephemeral=True,
            )
            return

        player.discord_user_id = member.id if member else None
        await self.db.update_tracked_player(player)

        if member:
            await interaction.response.send_message(
                f"Linked `{username}` to {member.mention}. They will be pinged on game notifications.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"Unlinked `{username}` from any Discord member.",
                ephemeral=True,
            )

    @app_commands.command(name="list", description="List all tracked players")
    async def list_players(self, interaction: discord.Interaction):
        """Show all tracked players for this server."""
        players = await self.db.get_tracked_players(interaction.guild_id)

        if not players:
            await interaction.response.send_message(
                "No players are being tracked. Use `/track` to add players.",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title="Tracked Players",
            color=discord.Color.blue(),
        )

        # Group by platform
        chesscom_players = [p for p in players if p.platform == Platform.CHESSCOM]
        lichess_players = [p for p in players if p.platform == Platform.LICHESS]

        def format_player(p):
            display = p.display_name or p.username
            if p.display_name:
                line = f"**{display}** ({p.username})"
            else:
                line = f"**{p.username}**"
            if p.discord_user_id:
                line += f" → <@{p.discord_user_id}>"
            return line

        if chesscom_players:
            lines = [format_player(p) for p in chesscom_players]
            embed.add_field(
                name="Chess.com",
                value="\n".join(lines),
                inline=False,
            )

        if lichess_players:
            lines = [format_player(p) for p in lichess_players]
            embed.add_field(
                name="Lichess",
                value="\n".join(lines),
                inline=False,
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="playerstats", description="Show recent stats for a player")
    @app_commands.describe(
        platform="The chess platform",
        username="The player's username",
    )
    async def playerstats(
        self,
        interaction: discord.Interaction,
        platform: PlatformChoice,
        username: str,
    ):
        """Show recent statistics for a tracked player."""
        await interaction.response.defer()

        # Find the player
        player = await self.db.get_tracked_player(
            interaction.guild_id, platform, username
        )

        if not player:
            await interaction.followup.send(
                f"`{username}` is not being tracked. Use `/track` first.",
            )
            return

        # Get today's stats
        stats = await self.db.get_daily_stats(player.id)

        display_name = player.display_name or player.username
        platform_name = format_platform_name(platform)

        embed = discord.Embed(
            title=f"Today's Stats: {display_name}",
            description=f"Platform: {platform_name}",
            color=discord.Color.blue(),
        )

        if stats["total_games"] == 0:
            embed.add_field(
                name="Activity",
                value="No games played today",
                inline=False,
            )
        else:
            embed.add_field(
                name="Games",
                value=f"{stats['total_games']} total",
                inline=True,
            )
            embed.add_field(
                name="Record",
                value=f"{stats['wins']}W / {stats['losses']}L / {stats['draws']}D",
                inline=True,
            )

            # Time control breakdown
            for tc, tc_stats in stats["by_time_control"].items():
                if tc_stats["games"] > 0:
                    change = tc_stats["rating_change"]
                    change_str = f"+{change}" if change > 0 else str(change)
                    embed.add_field(
                        name=tc.capitalize(),
                        value=f"{tc_stats['games']} games | {tc_stats['final_rating']} ({change_str})",
                        inline=True,
                    )

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="refresh", description="Manually check for new games now")
    async def refresh(self, interaction: discord.Interaction):
        """Manually trigger a refresh of all tracked players."""
        await interaction.response.defer(ephemeral=True)

        players = await self.db.get_tracked_players(interaction.guild_id)
        if not players:
            await interaction.followup.send(
                "No players are being tracked. Use `/track` to add players.",
                ephemeral=True,
            )
            return

        new_games = await self.tracker.refresh_guild(interaction.guild_id)

        if new_games > 0:
            await interaction.followup.send(
                f"Refresh complete! Found **{new_games}** new game(s).",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "Refresh complete. No new games found.",
                ephemeral=True,
            )

    @app_commands.command(name="history", description="Show the last 5 games for a player")
    @app_commands.describe(
        platform="The chess platform",
        username="The player's username",
    )
    async def history(
        self,
        interaction: discord.Interaction,
        platform: PlatformChoice,
        username: str,
    ):
        """Show the last 5 games for a tracked player."""
        await interaction.response.defer()

        # Find the player
        player = await self.db.get_tracked_player(
            interaction.guild_id, platform, username
        )

        if not player:
            await interaction.followup.send(
                f"`{username}` is not being tracked. Use `/track` first.",
            )
            return

        # Get recent games
        games = await self.db.get_recent_games(player.id, limit=5)

        display_name = player.display_name or player.username
        platform_name = format_platform_name(platform)

        if not games:
            await interaction.followup.send(
                f"No games recorded yet for **{display_name}** on {platform_name}."
            )
            return

        # Send header message
        await interaction.followup.send(
            f"**Recent Games for {display_name}** ({platform_name}) - Last {len(games)} games:"
        )

        # Send each game as a separate message with board image (oldest to newest)
        for game in reversed(games):
            embed, file = self._create_game_embed(player, game)
            await interaction.followup.send(embed=embed, file=file)

    @app_commands.command(name="video", description="Generate a video replay of a game")
    @app_commands.describe(
        platform="The chess platform",
        username="The player's username",
        game_number="Which recent game (1=most recent, 2=second most recent, etc.)",
    )
    async def video(
        self,
        interaction: discord.Interaction,
        platform: PlatformChoice,
        username: str,
        game_number: int = 1,
    ):
        """Generate a video replay of a player's game with evaluation bar."""
        await interaction.response.defer()

        # Validate game_number
        if game_number < 1 or game_number > 10:
            await interaction.followup.send(
                "Game number must be between 1 and 10.",
            )
            return

        # Find the player
        player = await self.db.get_tracked_player(
            interaction.guild_id, platform, username
        )

        if not player:
            await interaction.followup.send(
                f"`{username}` is not being tracked. Use `/track` first.",
            )
            return

        # Get recent games
        games = await self.db.get_recent_games(player.id, limit=game_number)

        if not games or len(games) < game_number:
            await interaction.followup.send(
                f"Not enough games recorded for **{username}**. "
                f"Only {len(games) if games else 0} game(s) available.",
            )
            return

        game = games[game_number - 1]

        # Fetch PGN from API
        await interaction.followup.send(
            f"Fetching game and generating video... This may take a moment."
        )

        pgn = None
        if platform == "lichess":
            pgn = await self.tracker.lichess_client.get_game_pgn(game.game_id)
        else:
            pgn = await self.tracker.chesscom_client.get_game_pgn(
                username, game.game_id
            )

        if not pgn:
            await interaction.followup.send(
                "Could not fetch the game PGN. The game may be too old or unavailable.",
            )
            return

        # Generate video
        video_bytes = await generate_game_video_async(pgn)

        if not video_bytes:
            await interaction.followup.send(
                "Failed to generate video. Please try again later.",
            )
            return

        # Send video
        display_name = player.display_name or player.username
        video_file = discord.File(
            io.BytesIO(video_bytes),
            filename=f"{display_name}_{game.game_id}.mp4",
        )

        await interaction.followup.send(
            f"**{display_name}** vs **{game.opponent}** ({game.time_control_display})",
            file=video_file,
        )

    def _create_game_embed(self, player: TrackedPlayer, game) -> tuple[discord.Embed, discord.File | None]:
        """Create an embed and board image for a game."""
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
            f"**Format:** {tc_emoji} {game.time_control.capitalize()} ({game.time_control_display})\n"
            f"**Rating:** {game.rating_after} ({rating_change_str})\n"
            f"**Played:** {time_str}"
        )

        # Add board image
        file = None
        if game.final_fen:
            file = get_board_discord_file(game.final_fen, f"board_{game.game_id}.png")
            if file:
                embed.set_image(url=f"attachment://board_{game.game_id}.png")

        return embed, file


async def setup(bot: commands.Bot):
    """Setup function called when loading the cog."""
    # This will be called with proper dependencies from bot.py
    pass
