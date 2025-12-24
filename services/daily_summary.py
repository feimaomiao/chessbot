import logging
from datetime import datetime, timedelta

import discord
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from database import DatabaseManager
from utils.helpers import format_platform_name, format_rating_change

logger = logging.getLogger(__name__)

# Summary interval in minutes
SUMMARY_INTERVAL_MINUTES = 10


class SummaryService:
    """Service for generating and sending periodic summaries."""

    def __init__(self, bot, db: DatabaseManager):
        self.bot = bot
        self.db = db
        self.scheduler = AsyncIOScheduler()
        self._job = None

    async def start(self):
        """Start the scheduler for periodic summaries."""
        # Schedule summary job every 10 minutes
        self._job = self.scheduler.add_job(
            self._send_all_summaries,
            IntervalTrigger(minutes=SUMMARY_INTERVAL_MINUTES),
            id="periodic_summary",
        )

        self.scheduler.start()
        logger.info(f"Summary service started (every {SUMMARY_INTERVAL_MINUTES} minutes)")

    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("Summary service stopped")

    async def _send_all_summaries(self):
        """Send summaries to all guilds with configured channels."""
        guilds = await self.db.get_all_guilds()

        for guild in guilds:
            if guild.summary_channel_id:
                await self._send_guild_summary(guild.id)

    async def _send_guild_summary(self, guild_id: int):
        """Generate and send summary for a guild."""
        import discord

        guild = await self.db.get_or_create_guild(guild_id)

        if not guild.summary_channel_id:
            return

        # Try cache first, then fetch
        channel = self.bot.get_channel(guild.summary_channel_id)
        if not channel:
            try:
                channel = await self.bot.fetch_channel(guild.summary_channel_id)
            except discord.NotFound:
                logger.warning(f"Summary channel {guild.summary_channel_id} not found")
                return
            except discord.Forbidden:
                logger.warning(f"No access to summary channel {guild.summary_channel_id}")
                return

        # Get all tracked players for this guild
        players = await self.db.get_tracked_players(guild_id)

        if not players:
            return

        # Generate summary for the last interval period
        embed = await self._generate_summary_embed(players)

        # Only send if there's activity
        if embed:
            try:
                await channel.send(embed=embed)
                logger.info(f"Sent summary for guild {guild_id}")
            except Exception as e:
                logger.error(f"Failed to send summary: {e}")

    async def _generate_summary_embed(self, players) -> discord.Embed | None:
        """Generate the summary embed for recent activity."""
        now = datetime.utcnow()
        since = now - timedelta(minutes=SUMMARY_INTERVAL_MINUTES)

        embed = discord.Embed(
            title="Chess Activity Update",
            description=f"Games in the last {SUMMARY_INTERVAL_MINUTES} minutes",
            color=discord.Color.blue(),
            timestamp=now,
        )

        has_activity = False

        for player in players:
            # Get games from the last interval
            games = await self.db.get_player_games_since(player.id, since)

            if not games:
                continue

            has_activity = True
            display_name = player.display_name or player.username
            platform_name = format_platform_name(player.platform)

            # Calculate stats for this period
            wins = sum(1 for g in games if g.result == "win")
            losses = sum(1 for g in games if g.result == "loss")
            draws = sum(1 for g in games if g.result == "draw")
            total = len(games)

            # Get rating changes by time control
            tc_stats = {}
            for game in games:
                tc = game.time_control
                if tc not in tc_stats:
                    tc_stats[tc] = {"rating_change": 0, "final_rating": 0}
                tc_stats[tc]["rating_change"] += game.rating_change
                tc_stats[tc]["final_rating"] = game.rating_after

            # Build player summary
            lines = [f"Games: {total} ({wins}W / {losses}L / {draws}D)"]

            for tc, stats in tc_stats.items():
                rating_change = format_rating_change(stats["rating_change"])
                lines.append(f"{tc.capitalize()}: {stats['final_rating']} ({rating_change})")

            embed.add_field(
                name=f"{display_name} ({platform_name})",
                value="\n".join(lines),
                inline=False,
            )

        if not has_activity:
            # Don't send if no activity
            return None

        return embed

    async def send_manual_summary(self, guild_id: int):
        """Generate an on-demand summary embed."""
        players = await self.db.get_tracked_players(guild_id)

        if not players:
            return None

        # For manual summary, show today's full stats
        embed = await self._generate_daily_summary_embed(players)
        return embed

    async def _generate_daily_summary_embed(self, players) -> discord.Embed:
        """Generate a summary embed for the last 24 hours."""
        embed = discord.Embed(
            title="Chess Summary - Last 24 Hours",
            color=discord.Color.blue(),
        )

        has_activity = False

        for player in players:
            stats = await self.db.get_stats_last_24h(player.id)

            if stats["total_games"] == 0:
                continue

            has_activity = True
            display_name = player.display_name or player.username
            platform_name = format_platform_name(player.platform)

            lines = [
                f"Games: {stats['total_games']} "
                f"({stats['wins']}W / {stats['losses']}L / {stats['draws']}D)"
            ]

            for tc, tc_stats in stats["by_time_control"].items():
                if tc_stats["games"] > 0:
                    rating_change = format_rating_change(tc_stats["rating_change"])
                    lines.append(
                        f"{tc.capitalize()}: {tc_stats['final_rating']} ({rating_change})"
                    )

            embed.add_field(
                name=f"{display_name} ({platform_name})",
                value="\n".join(lines),
                inline=False,
            )

        if not has_activity:
            embed.description = "No games played in the last 24 hours by tracked players."

        return embed
