import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import Platform
from database import DatabaseManager
from services.daily_summary import SummaryService

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

    @setchannel.error
    @manual_summary.error
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
