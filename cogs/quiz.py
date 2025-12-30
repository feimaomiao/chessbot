"""Commands for chess quizzes."""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from database import DatabaseManager
from services.tracker import GameTracker
from services.quiz_service import QuizService

logger = logging.getLogger(__name__)


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
    async def quiz(
        self,
        interaction: discord.Interaction,
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

        # Check for tracked players
        players = await self.db.get_tracked_players(interaction.guild_id)
        if not players:
            await interaction.followup.send(
                "No tracked players in this server. "
                "Use `/track` to add players first.",
                ephemeral=True,
            )
            return

        # Start the quiz
        success = await self.quiz_service.start_quiz(interaction)

        if not success:
            await interaction.followup.send(
                "Could not generate a quiz puzzle. "
                "This might happen if tracked players don't have any lost games "
                "(by checkmate or resignation) with blunders (300+ centipawn loss), "
                "or if Stockfish is unavailable. Try again!",
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
