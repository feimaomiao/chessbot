import asyncio
import logging
from datetime import datetime

import discord
from discord import app_commands
from discord.ext import commands

from api import ChessComClient, LichessClient
from config import Platform
from database import DatabaseManager
from services.daily_summary import SummaryService
from utils.accuracy import calculate_accuracy_from_pgn
from utils.stats import get_stats_tracker, get_system_specs
from utils.video import get_stockfish_evaluator

logger = logging.getLogger(__name__)


class AdminCog(commands.Cog):
    """Admin commands for bot configuration."""

    def __init__(
        self,
        bot: commands.Bot,
        db: DatabaseManager,
        summary_service: SummaryService,
    ):
        self.bot = bot
        self.db = db
        self.summary_service = summary_service

    @app_commands.command(name="setchannel", description="Set this channel for game notifications and daily summaries")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def setchannel(self, interaction: discord.Interaction):
        """Set the current channel for both notifications and summaries."""
        guild = await self.db.get_or_create_guild(interaction.guild_id)
        channel = interaction.channel

        guild.notification_channel_id = channel.id
        guild.summary_channel_id = channel.id
        await self.db.update_guild(guild)

        await interaction.response.send_message(
            f"Game notifications and daily summaries (at 00:00 UTC) will now be sent to this channel.",
            ephemeral=True,
        )

    @app_commands.command(name="status", description="Show bot status and configuration")
    async def status(self, interaction: discord.Interaction):
        """Show bot status and current configuration."""
        await interaction.response.defer(ephemeral=True)

        guild = await self.db.get_or_create_guild(interaction.guild_id)
        players = await self.db.get_tracked_players(interaction.guild_id)

        embed = discord.Embed(
            title="Chess Tracker Status",
            color=discord.Color.green(),
        )

        # Channel configuration - try cache first, then fetch
        if guild.notification_channel_id:
            channel = self.bot.get_channel(guild.notification_channel_id)
            if not channel:
                try:
                    channel = await self.bot.fetch_channel(guild.notification_channel_id)
                except discord.NotFound:
                    logger.warning(f"Notification channel {guild.notification_channel_id} not found")
                    channel = None
                except discord.Forbidden:
                    logger.warning(f"No access to notification channel {guild.notification_channel_id}")
                    channel = None
                except Exception as e:
                    logger.error(f"Error fetching notification channel: {e}")
                    channel = None
            notif_text = channel.mention if channel else f"<#{guild.notification_channel_id}>"
        else:
            notif_text = "Not set - use `/setchannel`"

        if guild.summary_channel_id:
            channel = self.bot.get_channel(guild.summary_channel_id)
            if not channel:
                try:
                    channel = await self.bot.fetch_channel(guild.summary_channel_id)
                except discord.NotFound:
                    logger.warning(f"Summary channel {guild.summary_channel_id} not found")
                    channel = None
                except discord.Forbidden:
                    logger.warning(f"No access to summary channel {guild.summary_channel_id}")
                    channel = None
                except Exception as e:
                    logger.error(f"Error fetching summary channel: {e}")
                    channel = None
            summary_text = channel.mention if channel else f"<#{guild.summary_channel_id}>"
        else:
            summary_text = "Not set - use `/setchannel`"

        embed.add_field(
            name="Notification Channel",
            value=notif_text,
            inline=True,
        )
        embed.add_field(
            name="Summary Channel",
            value=summary_text,
            inline=True,
        )
        embed.add_field(
            name="Poll Interval",
            value="Every 60 seconds",
            inline=True,
        )

        # Tracked players list
        if players:
            chesscom = [p for p in players if p.platform == Platform.CHESSCOM]
            lichess = [p for p in players if p.platform == Platform.LICHESS]

            player_lines = []
            if chesscom:
                names = [p.display_name or p.username for p in chesscom]
                player_lines.append(f"**Chess.com:** {', '.join(names)}")
            if lichess:
                names = [p.display_name or p.username for p in lichess]
                player_lines.append(f"**Lichess:** {', '.join(names)}")

            embed.add_field(
                name=f"Tracked Players ({len(players)})",
                value="\n".join(player_lines),
                inline=False,
            )
        else:
            embed.add_field(
                name="Tracked Players",
                value="None - use `/track` to add players",
                inline=False,
            )

        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="summary", description="Send a summary now")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def manual_summary(self, interaction: discord.Interaction):
        """Manually trigger a daily summary."""
        await interaction.response.defer()

        embed = await self.summary_service.send_manual_summary(interaction.guild_id)

        if embed:
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send(
                "No tracked players found. Add players with `/track` first.",
                ephemeral=True,
            )

    @app_commands.command(name="specs", description="Show system specs and performance statistics")
    async def specs(self, interaction: discord.Interaction):
        """Show system specifications and performance stats."""
        try:
            await interaction.response.defer(ephemeral=True)
        except discord.NotFound:
            # Interaction expired before we could respond
            logger.warning("Interaction expired for /specs command")
            return

        # Get system specs
        specs = get_system_specs()

        # Get performance stats
        tracker = get_stats_tracker()
        perf_stats = tracker.get_stats()

        # Check Stockfish availability
        evaluator = get_stockfish_evaluator()
        stockfish_status = "Available" if evaluator.available else "Not available"

        embed = discord.Embed(
            title="System Specifications & Performance",
            color=discord.Color.blue(),
        )

        # System info
        system_info = (
            f"**Platform:** {specs.get('platform', 'Unknown')} ({specs.get('architecture', 'Unknown')})\n"
            f"**CPU:** {specs.get('processor', 'Unknown')}\n"
            f"**Cores:** {specs.get('cpu_cores_physical', '?')} physical / {specs.get('cpu_cores_logical', '?')} logical\n"
            f"**Memory:** {specs.get('memory_available_gb', '?')} GB available / {specs.get('memory_total_gb', '?')} GB total\n"
            f"**Python:** {specs.get('python_version', 'Unknown')}\n"
            f"**Stockfish:** {stockfish_status}"
        )
        embed.add_field(name="System", value=system_info, inline=False)

        # Evaluation stats
        eval_stats = perf_stats.evaluation
        if eval_stats.total_operations > 0:
            eval_info = (
                f"**Avg:** {eval_stats.avg_time_per_position_ms:.1f}ms/position\n"
                f"**Total:** {eval_stats.total_positions} positions in {eval_stats.total_operations} operations"
            )
            embed.add_field(name="Evaluation Performance", value=eval_info, inline=False)

        # Quiz evaluation stats
        quiz_stats = perf_stats.quiz_evaluation
        if quiz_stats.total_operations > 0:
            quiz_info = (
                f"**Avg:** {quiz_stats.avg_time_per_position_ms:.1f}ms/position\n"
                f"**Total:** {quiz_stats.total_positions} positions in {quiz_stats.total_operations} quizzes"
            )
            embed.add_field(name="Quiz Evaluation Performance", value=quiz_info, inline=False)

        # Video generation stats
        video_stats = perf_stats.video_generation
        if video_stats.total_operations > 0:
            video_info = (
                f"**Avg:** {video_stats.avg_time_per_position_ms:.1f}ms/position\n"
                f"**Total:** {video_stats.total_positions} positions in {video_stats.total_operations} videos"
            )
            embed.add_field(name="Video Generation Performance", value=video_info, inline=False)

        # Cache stats
        if perf_stats.total_cache_hits > 0 or perf_stats.total_cache_misses > 0:
            cache_info = (
                f"**Hit Rate:** {perf_stats.cache_hit_rate:.1f}%\n"
                f"**Hits:** {perf_stats.total_cache_hits} / **Misses:** {perf_stats.total_cache_misses}"
            )
            embed.add_field(name="Evaluation Cache", value=cache_info, inline=False)

        # Quiz cache stats
        if perf_stats.quiz_cache_hits > 0 or perf_stats.quiz_cache_misses > 0:
            quiz_cache_info = (
                f"**Hit Rate:** {perf_stats.quiz_cache_hit_rate:.1f}%\n"
                f"**Hits:** {perf_stats.quiz_cache_hits} / **Misses:** {perf_stats.quiz_cache_misses}"
            )
            embed.add_field(name="Quiz Evaluation Cache", value=quiz_cache_info, inline=False)

        # Stats tracking info
        if perf_stats.started_at:
            try:
                started = datetime.fromisoformat(perf_stats.started_at)
                embed.set_footer(text=f"Stats tracking since {started.strftime('%Y-%m-%d %H:%M UTC')}")
            except ValueError:
                pass

        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="reprocess", description="Reprocess all games to calculate accuracy (admin only)")
    @app_commands.checks.has_permissions(administrator=True)
    async def reprocess(self, interaction: discord.Interaction):
        """Reprocess all games without accuracy to calculate their accuracy scores."""
        try:
            await interaction.response.defer(ephemeral=True)
        except discord.NotFound:
            logger.warning("Interaction expired for /reprocess command")
            return

        # Check Stockfish availability
        evaluator = get_stockfish_evaluator()
        if not evaluator.available:
            await interaction.followup.send(
                "Stockfish is not available. Cannot calculate accuracy.",
                ephemeral=True,
            )
            return

        # Get all games without accuracy
        games = await self.db.get_games_without_accuracy()

        if not games:
            await interaction.followup.send(
                "All games already have accuracy calculated!",
                ephemeral=True,
            )
            return

        await interaction.followup.send(
            f"Found **{len(games)}** games without accuracy. Starting reprocessing...\n"
            f"This may take a while. Progress updates will be sent here.",
            ephemeral=True,
        )

        # Create API clients
        lichess_client = LichessClient()
        chesscom_client = ChessComClient()

        # Get player info for fetching PGNs
        players_cache = {}

        processed = 0
        failed = 0
        skipped = 0

        try:
            for i, game in enumerate(games):
                # Get player info (cached)
                if game.player_id not in players_cache:
                    player = await self.db.get_tracked_player_by_id(game.player_id)
                    players_cache[game.player_id] = player

                player = players_cache.get(game.player_id)
                if not player:
                    skipped += 1
                    continue

                try:
                    # Fetch PGN
                    pgn = None
                    if game.platform == Platform.LICHESS:
                        pgn = await lichess_client.get_game_pgn(game.game_id)
                    elif game.platform == Platform.CHESSCOM:
                        pgn = await chesscom_client.get_game_pgn(player.username, game.game_id)

                    if not pgn:
                        skipped += 1
                        continue

                    # Calculate accuracy
                    accuracy = await calculate_accuracy_from_pgn(pgn, game.player_color)

                    if accuracy is not None:
                        await self.db.update_game_accuracy(game.id, accuracy)
                        processed += 1
                    else:
                        skipped += 1

                except Exception as e:
                    logger.error(f"Error processing game {game.game_id}: {e}")
                    failed += 1

                # Progress update every 10 games
                if (i + 1) % 10 == 0:
                    await interaction.followup.send(
                        f"Progress: {i + 1}/{len(games)} games processed...",
                        ephemeral=True,
                    )

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

        finally:
            await lichess_client.close()
            await chesscom_client.close()

        await interaction.followup.send(
            f"**Reprocessing complete!**\n"
            f"Processed: {processed}\n"
            f"Skipped: {skipped}\n"
            f"Failed: {failed}",
            ephemeral=True,
        )

    @setchannel.error
    @manual_summary.error
    @reprocess.error
    async def command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ):
        """Handle command errors."""
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message(
                "You need the **Manage Server** permission to use this command.",
                ephemeral=True,
            )
        elif isinstance(error, app_commands.TransformerError):
            await interaction.response.send_message(
                "Invalid channel. Please select a channel from the dropdown, not type its name.",
                ephemeral=True,
            )
        else:
            logger.error(f"Command error: {error}")
            await interaction.response.send_message(
                "An error occurred. Please try again.",
                ephemeral=True,
            )


async def setup(bot: commands.Bot):
    """Setup function called when loading the cog."""
    pass
