import io
import logging
from typing import Literal

import discord
from discord import app_commands
from discord.ext import commands

from config import Platform
from database import DatabaseManager, TrackedPlayer
from services.tracker import GameTracker
from services.analysis import analyze_games, calculate_accuracy_stats, AnalysisResult
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

        new_games, backfilled = await self.tracker.refresh_guild(interaction.guild_id)

        # Build response message
        parts = []
        if new_games > 0:
            parts.append(f"**{new_games}** new game(s)")
        if backfilled > 0:
            parts.append(f"**{backfilled}** historical game(s) backfilled")

        if parts:
            await interaction.followup.send(
                f"Refresh complete! {', '.join(parts)}.",
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
        """Show the last 5 games for a tracked player with video replays."""
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
            f"**Recent Games for {display_name}** ({platform_name}) - Last {len(games)} games:\n"
            f"Generating video replays... This may take a moment."
        )

        # Send each game as a separate message with video (oldest to newest)
        for game in reversed(games):
            # Fetch PGN from API
            pgn = None
            if platform == "lichess":
                pgn = await self.tracker.lichess_client.get_game_pgn(game.game_id)
            else:
                pgn = await self.tracker.chesscom_client.get_game_pgn(
                    username, game.game_id
                )

            if pgn:
                # Generate video from tracked player's perspective
                video_bytes = await generate_game_video_async(pgn, player_color=game.player_color)

                if video_bytes:
                    # Create embed with game info
                    embed = self._create_history_embed(player, game)
                    video_file = discord.File(
                        io.BytesIO(video_bytes),
                        filename=f"{display_name}_{game.game_id}.mp4",
                    )
                    await interaction.followup.send(embed=embed, file=video_file)
                    continue

            # Fallback to static board image if video generation fails
            embed, file = self._create_game_embed(player, game)
            await interaction.followup.send(embed=embed, file=file)

    def _create_history_embed(self, player: TrackedPlayer, game) -> discord.Embed:
        """Create an embed for game history (without board image, for use with video)."""
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
        description_lines = [
            f"**Format:** {tc_emoji} {game.time_control.capitalize()} ({game.time_control_display})",
            f"**Rating:** {game.rating_after} ({rating_change_str})",
        ]

        if game.accuracy is not None:
            description_lines.append(f"**Accuracy:** {game.accuracy:.1f}%")

        description_lines.append(f"**Played:** {time_str}")
        embed.description = "\n".join(description_lines)

        return embed

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

        # Generate video from tracked player's perspective
        video_bytes = await generate_game_video_async(pgn, player_color=game.player_color)

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

    @app_commands.command(name="analyze", description="Analyze a player's recent games and openings")
    @app_commands.describe(
        platform="The chess platform",
        username="The player's username",
    )
    async def analyze(
        self,
        interaction: discord.Interaction,
        platform: PlatformChoice,
        username: str,
    ):
        """Analyze a player's recent games for opening stats and performance."""
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

        # Send initial status message
        await interaction.followup.send(
            f"Fetching games for **{username}**... This may take a moment."
        )

        # Fetch games from API
        if platform == "lichess":
            games = await self.tracker.lichess_client.get_games_for_analysis(username)
        else:
            games = await self.tracker.chesscom_client.get_games_for_analysis(username)

        if not games:
            await interaction.followup.send(
                f"No games found for **{username}** on {format_platform_name(platform)}."
            )
            return

        # Run analysis on API games (for openings)
        result = analyze_games(games)

        # Get accuracy stats from database games
        db_games = await self.db.get_recent_games(player.id, limit=1000)
        if db_games:
            avg_acc, acc_count = calculate_accuracy_stats(db_games)
            result.avg_accuracy = avg_acc
            result.games_with_accuracy = acc_count

        # Create and send embed
        embed = self._create_analysis_embed(player, platform, result)
        await interaction.followup.send(embed=embed)

    def _create_analysis_embed(
        self,
        player: TrackedPlayer,
        platform: str,
        result: AnalysisResult
    ) -> discord.Embed:
        """Create a Discord embed for analysis results."""
        display_name = player.display_name or player.username
        platform_name = format_platform_name(platform)

        # Date range formatting
        if result.date_range_start and result.date_range_end:
            date_range = (
                f"{result.date_range_start.strftime('%b %Y')} - "
                f"{result.date_range_end.strftime('%b %Y')}"
            )
        else:
            date_range = "N/A"

        embed = discord.Embed(
            title=f"Account Analysis: {display_name}",
            description=(
                f"**Platform:** {platform_name} | "
                f"**Games:** {result.total_games} | "
                f"**Period:** {date_range}"
            ),
            color=discord.Color.blue(),
        )

        # Rating progression
        if result.starting_rating > 0:
            change = result.rating_change
            change_str = f"+{change}" if change > 0 else str(change)
            embed.add_field(
                name="Rating Progression",
                value=(
                    f"{result.starting_rating} -> {result.ending_rating} ({change_str})\n"
                    f"High: {result.rating_high} | Low: {result.rating_low}"
                ),
                inline=False,
            )

        # Performance by color
        if result.white_games > 0 or result.black_games > 0:
            white_total = result.white_games
            white_wr = f"{result.white_win_rate:.0f}%" if white_total > 0 else "N/A"
            white_ld = f"{result.white_losses}L/{result.white_draws}D" if white_total > 0 else ""

            black_total = result.black_games
            black_wr = f"{result.black_win_rate:.0f}%" if black_total > 0 else "N/A"
            black_ld = f"{result.black_losses}L/{result.black_draws}D" if black_total > 0 else ""

            embed.add_field(
                name="Performance by Color",
                value=(
                    f"White: {white_total} games ({white_wr} win, {white_ld})\n"
                    f"Black: {black_total} games ({black_wr} win, {black_ld})"
                ),
                inline=False,
            )

        # Accuracy stats
        if result.avg_accuracy is not None:
            embed.add_field(
                name="Average Accuracy",
                value=f"{result.avg_accuracy:.1f}% ({result.games_with_accuracy} games analyzed)",
                inline=False,
            )

        # Termination breakdown (only show non-zero values)
        termination_parts = []
        if result.termination_checkmate > 0:
            termination_parts.append(f"Checkmate: {result.termination_checkmate}")
        if result.termination_resign > 0:
            termination_parts.append(f"Resignation: {result.termination_resign}")
        if result.termination_timeout > 0:
            termination_parts.append(f"Timeout: {result.termination_timeout}")
        if result.termination_stalemate > 0:
            termination_parts.append(f"Stalemate: {result.termination_stalemate}")
        if result.termination_repetition > 0:
            termination_parts.append(f"Repetition: {result.termination_repetition}")
        if result.termination_agreed > 0:
            termination_parts.append(f"Draw Agreement: {result.termination_agreed}")

        if termination_parts:
            embed.add_field(
                name="Game Endings",
                value=" | ".join(termination_parts),
                inline=False,
            )

        # Top openings as white
        if result.top_openings_white:
            lines = []
            for i, opening in enumerate(result.top_openings_white[:5], 1):
                eco = f"({opening.eco})" if opening.eco else ""
                lines.append(
                    f"{i}. {opening.name} {eco} - {opening.games} games, {opening.win_rate:.0f}% win"
                )
            embed.add_field(
                name="Top Openings as White",
                value="\n".join(lines),
                inline=False,
            )

        # Top openings as black
        if result.top_openings_black:
            lines = []
            for i, opening in enumerate(result.top_openings_black[:5], 1):
                eco = f"({opening.eco})" if opening.eco else ""
                lines.append(
                    f"{i}. {opening.name} {eco} - {opening.games} games, {opening.win_rate:.0f}% win"
                )
            embed.add_field(
                name="Top Openings as Black",
                value="\n".join(lines),
                inline=False,
            )

        return embed

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
        description_lines = [
            f"**Format:** {tc_emoji} {game.time_control.capitalize()} ({game.time_control_display})",
            f"**Rating:** {game.rating_after} ({rating_change_str})",
        ]

        if game.accuracy is not None:
            description_lines.append(f"**Accuracy:** {game.accuracy:.1f}%")

        description_lines.append(f"**Played:** {time_str}")
        embed.description = "\n".join(description_lines)

        # Add board image (from tracked player's perspective)
        file = None
        if game.final_fen:
            flipped = game.player_color == "black"
            file = get_board_discord_file(game.final_fen, f"board_{game.game_id}.png", flipped=flipped)
            if file:
                embed.set_image(url=f"attachment://board_{game.game_id}.png")

        return embed, file


async def setup(bot: commands.Bot):
    """Setup function called when loading the cog."""
    # This will be called with proper dependencies from bot.py
    pass
