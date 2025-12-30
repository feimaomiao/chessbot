"""Commands for chess quizzes."""

import logging
from typing import Literal, Optional

import discord
from discord import app_commands
from discord.ext import commands

from database import DatabaseManager
from services.tracker import GameTracker
from services.quiz_service import QuizService

logger = logging.getLogger(__name__)

PlatformChoice = Literal["chesscom", "lichess"]


class QuizCog(commands.Cog):
    """Commands for chess quizzes based on tracked players' games."""

    def __init__(
        self,
        bot: commands.Bot,
        db: DatabaseManager,
        tracker: GameTracker,
        quiz_service: QuizService,
    ):
        self.bot = bot
        self.db = db
        self.tracker = tracker
        self.quiz_service = quiz_service

    @app_commands.command(
        name="quiz",
        description="Start a chess puzzle from a tracked player's game",
    )
    @app_commands.describe(
        platform="Filter by chess platform (optional)",
        username="Filter by specific player username (optional)",
        global_search="Search across all tracked players from all servers (default: False)",
    )
    async def quiz(
        self,
        interaction: discord.Interaction,
        platform: Optional[PlatformChoice] = None,
        username: Optional[str] = None,
        global_search: bool = False,
    ):
        """Start a chess quiz puzzle."""
        await interaction.response.defer()

        # Check for active quiz
        if await self.quiz_service.is_quiz_active(interaction.channel_id):
            await interaction.followup.send(
                "A quiz is already active in this channel! "
                "Use `/answer` to submit your answer or `/reveal` to see the solution.",
                ephemeral=True,
            )
            return

        # If platform and username provided, find the specific player
        player_id = None
        if platform and username:
            # First check in current guild
            player = await self.db.get_tracked_player(
                interaction.guild_id, platform, username
            )
            # If not found and global search enabled, search globally
            if not player and global_search:
                # Search across all guilds for this player
                all_players = await self.db.get_all_tracked_players()
                for p in all_players:
                    if p.platform == platform and p.username.lower() == username.lower():
                        player = p
                        break

            if not player:
                await interaction.followup.send(
                    f"Player **{username}** on **{platform}** is not being tracked. "
                    f"Use `/track` to add them first.",
                    ephemeral=True,
                )
                return

            player_id = player.id
        elif platform or username:
            # Only one of platform/username provided
            await interaction.followup.send(
                "Please provide both `platform` and `username` to filter by a specific player.",
                ephemeral=True,
            )
            return

        # Check for tracked players (only for local search without specific player)
        if not global_search and not player_id:
            players = await self.db.get_tracked_players(interaction.guild_id)
            if not players:
                await interaction.followup.send(
                    "No tracked players in this server. "
                    "Use `/track` to add players first, or use `/quiz global_search:True` "
                    "to search across all servers.",
                    ephemeral=True,
                )
                return

        # Start the quiz
        success = await self.quiz_service.start_quiz(
            interaction, global_search=global_search, player_id=player_id
        )

        if not success:
            if player_id:
                await interaction.followup.send(
                    f"Could not generate a quiz puzzle for **{username}**. "
                    "They may not have any lost games (by checkmate or resignation) "
                    "with blunders (250+ centipawn loss), or Stockfish is unavailable.",
                    ephemeral=True,
                )
            elif global_search:
                await interaction.followup.send(
                    "Could not generate a quiz puzzle. "
                    "No tracked players across any server have lost games "
                    "(by checkmate or resignation) with blunders (250+ centipawn loss), "
                    "or Stockfish is unavailable. Try again!",
                    ephemeral=True,
                )
            else:
                await interaction.followup.send(
                    "Could not generate a quiz puzzle. "
                    "This might happen if tracked players don't have any lost games "
                    "(by checkmate or resignation) with blunders (250+ centipawn loss), "
                    "or if Stockfish is unavailable. Try `/quiz global_search:True` "
                    "to search across all servers.",
                    ephemeral=True,
                )

    @app_commands.command(
        name="answer",
        description="Submit your answer to the active quiz",
    )
    @app_commands.describe(
        move="Your answer in standard notation (e.g., Nf3, Qxd5, O-O)",
    )
    async def answer(
        self,
        interaction: discord.Interaction,
        move: str,
    ):
        """Submit an answer to the active quiz."""
        # Check if there's an active quiz
        if not await self.quiz_service.is_quiz_active(interaction.channel_id):
            await interaction.response.send_message(
                "No active quiz in this channel. Use `/quiz` to start one!",
                ephemeral=True,
            )
            return

        # Check the answer
        is_correct, message = await self.quiz_service.check_answer(
            interaction.channel_id,
            move,
            interaction.user.id,
            interaction.user.display_name,
        )

        if is_correct:
            # Winner announcement is handled by quiz_service.end_quiz
            await interaction.response.send_message(
                "Checking your answer...",
                ephemeral=True,
                delete_after=1,
            )
        elif message:
            await interaction.response.send_message(
                message,
                ephemeral=True,
            )
        else:
            # Shouldn't happen, but handle gracefully
            await interaction.response.send_message(
                "Something went wrong. Please try again.",
                ephemeral=True,
            )

    @app_commands.command(
        name="reveal",
        description="Reveal the answer to the active quiz",
    )
    async def reveal(
        self,
        interaction: discord.Interaction,
    ):
        """Reveal the answer to the active quiz."""
        # Check if there's an active quiz
        if not await self.quiz_service.is_quiz_active(interaction.channel_id):
            await interaction.response.send_message(
                "No active quiz in this channel. Use `/quiz` to start one!",
                ephemeral=True,
            )
            return

        # Reveal the answer - send ephemeral acknowledgment first
        await interaction.response.send_message(
            "Revealing answer...",
            ephemeral=True,
            delete_after=1,
        )

        correct_move = await self.quiz_service.reveal_answer(interaction.channel_id)

        if not correct_move:
            # Quiz may have been answered/revealed by someone else
            pass

    @app_commands.command(
        name="leaderboard",
        description="Show the quiz leaderboard for this server",
    )
    async def leaderboard(
        self,
        interaction: discord.Interaction,
    ):
        """Show the quiz leaderboard."""
        leaderboard = await self.db.get_quiz_leaderboard(interaction.guild_id)

        if not leaderboard:
            await interaction.response.send_message(
                "No quiz scores yet! Use `/quiz` to start playing.",
                ephemeral=True,
            )
            return

        # Build leaderboard embed
        embed = discord.Embed(
            title="Quiz Leaderboard",
            color=discord.Color.gold(),
        )

        # Format leaderboard entries
        entries = []
        medals = ["🥇", "🥈", "🥉"]
        for i, (user_id, username, score) in enumerate(leaderboard):
            rank = medals[i] if i < 3 else f"**{i + 1}.**"
            # Format score: show 2 decimal places for fractional, whole number otherwise
            score_str = f"{score:.2f}" if score % 1 != 0 else f"{int(score)}"
            entries.append(f"{rank} {username} - **{score_str}** points")

        embed.description = "\n".join(entries)

        # Show requester's rank if not in top 10
        user_in_top = any(user_id == interaction.user.id for user_id, _, _ in leaderboard)
        if not user_in_top:
            user_score = await self.db.get_user_quiz_score(
                interaction.guild_id, interaction.user.id
            )
            if user_score > 0:
                score_str = f"{user_score:.2f}" if user_score % 1 != 0 else f"{int(user_score)}"
                embed.set_footer(text=f"Your score: {score_str} points")

        await interaction.response.send_message(embed=embed)
