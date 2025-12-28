#!/usr/bin/env python3
"""
Chess Tracker Discord Bot

A bot that tracks chess.com and lichess.com players and notifies
when they complete games.
"""

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

import discord
from discord.ext import commands

from config import DATABASE_PATH, DISCORD_TOKEN, EVAL_CACHE_PATH, LOG_LEVEL
from database import DatabaseManager
from utils.video import get_eval_cache
from services.daily_summary import SummaryService
from services.notifications import NotificationService
from services.tracker import GameTracker
from cogs.tracking import TrackingCog
from cogs.admin import AdminCog

# Setup logging
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "chesstracker.log"

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL))

# Clear any existing handlers to prevent duplicates
root_logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5,
)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)


class ChessTrackerBot(commands.Bot):
    """Main bot class with service management."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.guilds = True

        super().__init__(
            command_prefix="!",  # Fallback, we use slash commands
            intents=intents,
        )

        self.db: DatabaseManager = None
        self.notification_service: NotificationService = None
        self.tracker: GameTracker = None
        self.summary_service: SummaryService = None

    async def setup_hook(self):
        """Initialize services and load cogs."""
        logger.info("Initializing services...")

        # Load evaluation cache from file
        eval_cache = get_eval_cache()
        eval_cache.load_from_file(EVAL_CACHE_PATH)

        # Initialize database
        self.db = DatabaseManager(DATABASE_PATH)
        await self.db.connect()
        logger.info("Database connected")

        # Initialize services
        self.notification_service = NotificationService(self, self.db)
        self.tracker = GameTracker(self.db, self.notification_service)
        self.summary_service = SummaryService(self, self.db)

        # Add cogs
        await self.add_cog(TrackingCog(self, self.db, self.tracker))
        await self.add_cog(AdminCog(self, self.db, self.summary_service))
        logger.info("Cogs loaded")

        # Sync commands globally
        logger.info("Syncing slash commands...")
        await self.tree.sync()
        logger.info("Slash commands synced")

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")

        # Sync commands to all connected guilds (instant, unlike global sync)
        for guild in self.guilds:
            try:
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                logger.info(f"Synced commands to guild: {guild.name}")
            except Exception as e:
                logger.error(f"Failed to sync to {guild.name}: {e}")

        # Start background services
        await self.tracker.start()
        await self.summary_service.start()

        # Set presence
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="chess games",
            )
        )

        logger.info("Bot is ready!")

    async def on_guild_join(self, guild: discord.Guild):
        """Called when the bot joins a new guild."""
        logger.info(f"Joined guild: {guild.name} (ID: {guild.id})")
        await self.db.get_or_create_guild(guild.id)

        # Sync commands to new guild immediately
        try:
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(f"Synced commands to new guild: {guild.name}")
        except Exception as e:
            logger.error(f"Failed to sync to {guild.name}: {e}")

    async def close(self):
        """Cleanup on shutdown."""
        logger.info("Shutting down...")

        # Save evaluation cache to file
        eval_cache = get_eval_cache()
        eval_cache.save_to_file(EVAL_CACHE_PATH)

        if self.tracker:
            await self.tracker.stop()

        if self.summary_service:
            self.summary_service.stop()

        if self.db:
            await self.db.close()

        await super().close()
        logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    bot = ChessTrackerBot()

    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
