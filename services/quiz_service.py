"""Quiz service for managing chess quizzes in Discord channels."""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import chess
import discord

from database import DatabaseManager, ActiveQuiz
from utils.board import get_board_discord_file
from utils.notation import parse_user_move
from utils.quiz import (
    find_missed_moves_async,
    select_quiz_position,
    classify_difficulty,
    get_continuation_moves_async,
)

if TYPE_CHECKING:
    from services.tracker import GameTracker

logger = logging.getLogger(__name__)

# Maximum number of game retries if no missed moves found
MAX_GAME_RETRIES = 10


def format_eval(centipawns: float) -> str:
    """
    Format centipawn evaluation for display.

    Args:
        centipawns: Evaluation in centipawns (positive = white advantage)

    Returns:
        Formatted string like "+1.50", "-0.30", or "M3" for mate
    """
    if centipawns >= 9900:
        mate_in = 10000 - int(centipawns)
        return f"M{mate_in}" if mate_in > 0 else "M1"
    elif centipawns <= -9900:
        mate_in = 10000 + int(centipawns)
        return f"-M{mate_in}" if mate_in > 0 else "-M1"
    else:
        pawns = centipawns / 100
        if pawns >= 0:
            return f"+{pawns:.2f}"
        else:
            return f"{pawns:.2f}"


def format_eval_comparison(eval_before: float, eval_after: float, player_color: str) -> str:
    """
    Format evaluation change with arrow, from the player's perspective.

    Args:
        eval_before: Evaluation before the move
        eval_after: Evaluation after the move
        player_color: "white" or "black"

    Returns:
        Formatted string like "+1.50 → +0.30 (-1.20)"
    """
    before_str = format_eval(eval_before)
    after_str = format_eval(eval_after)

    # Calculate the change from the player's perspective
    if player_color == "white":
        change = eval_after - eval_before
    else:
        change = eval_before - eval_after  # For black, lower eval is better

    change_pawns = change / 100
    if change >= 0:
        change_str = f"+{change_pawns:.2f}"
    else:
        change_str = f"{change_pawns:.2f}"

    return f"{before_str} → {after_str} ({change_str})"


class QuizService:
    """Manages active quizzes across Discord channels using database storage."""

    def __init__(self, bot, db: DatabaseManager, tracker: "GameTracker"):
        self.bot = bot
        self.db = db
        self.tracker = tracker

    async def is_quiz_active(self, channel_id: int) -> bool:
        """Check if a quiz is currently active in a channel."""
        quiz = await self.db.get_quiz(channel_id)
        return quiz is not None

    async def get_active_quiz(self, channel_id: int) -> Optional[ActiveQuiz]:
        """Get the active quiz for a channel, if any."""
        return await self.db.get_quiz(channel_id)

    async def start_quiz(
        self,
        interaction: discord.Interaction,
        global_search: bool = False,
        player_id: Optional[int] = None,
    ) -> bool:
        """
        Start a new quiz in the channel.

        Args:
            interaction: The Discord interaction
            global_search: If True, search across all tracked players globally
            player_id: If provided, only use games from this specific player

        Returns:
            True if quiz started successfully, False otherwise
        """
        if await self.is_quiz_active(interaction.channel_id):
            return False

        # Try to find a suitable game with missed moves
        for attempt in range(MAX_GAME_RETRIES):
            if player_id:
                result = await self.db.get_random_lost_game_for_player(player_id)
            elif global_search:
                result = await self.db.get_random_lost_game_global()
            else:
                result = await self.db.get_random_lost_game_with_player(interaction.guild_id)
            if not result:
                if player_id:
                    logger.info(f"No lost games found for player {player_id}")
                elif global_search:
                    logger.info("No lost games found globally")
                else:
                    logger.info(f"No lost games found in guild {interaction.guild_id}")
                return False

            player, game = result
            logger.info(
                f"Quiz attempt {attempt + 1}: Analyzing game {game.game_id} "
                f"({player.username} vs {game.opponent})"
            )

            # Fetch PGN
            pgn = await self._fetch_pgn(player, game)
            if not pgn:
                logger.warning(f"Could not fetch PGN for game {game.game_id}")
                continue

            # Find missed moves (blunders with 250+ centipawn loss)
            # Run in thread pool to avoid blocking Discord heartbeat
            missed_moves = await find_missed_moves_async(pgn, game.player_color, min_eval_loss=250)

            if not missed_moves:
                logger.info(f"No blunders (250+ cp) found in game {game.game_id}")
                continue

            # Select a quiz position
            selected = select_quiz_position(missed_moves)
            if not selected:
                continue

            # Found a suitable position!
            logger.info(
                f"Selected quiz position: move {selected.move_number}, "
                f"eval loss: {selected.eval_loss:.0f}cp, "
                f"best move: {selected.best_move_san}"
            )

            # Create and send the quiz
            return await self._send_quiz(interaction, player, game, selected)

        logger.info(f"Could not find suitable quiz game after {MAX_GAME_RETRIES} attempts")
        return False

    async def _send_quiz(
        self,
        interaction: discord.Interaction,
        player,
        game,
        missed_move,
    ) -> bool:
        """Send the quiz embed and save the quiz to the database."""
        board = chess.Board(missed_move.position_fen)
        whose_turn = "White" if board.turn == chess.WHITE else "Black"
        player_display = player.display_name or player.username
        difficulty = classify_difficulty(missed_move.eval_loss)

        # Create embed
        embed = discord.Embed(
            title="Chess Quiz!",
            description=(
                f"**{player_display}** blundered in their game "
                f"against **{game.opponent}**.\n\n"
                f"They played **{missed_move.player_move_san}** - what was the best move?\n"
                f"Use `/answer <move>` to submit (e.g., `/answer Nf3`)\n"
                f"Use `/reveal` to see the answer"
            ),
            color=discord.Color.blue(),
        )

        embed.add_field(name="Move", value=f"#{missed_move.move_number}", inline=True)
        embed.add_field(name="Difficulty", value=difficulty.capitalize(), inline=True)
        embed.add_field(name="To Play", value=whose_turn, inline=True)
        embed.add_field(name="Eval", value=format_eval(missed_move.eval_before), inline=True)

        embed.set_image(url="attachment://quiz_position.png")

        # Generate board image (flipped if black to play from black's perspective)
        flipped = board.turn == chess.BLACK
        board_file = get_board_discord_file(
            missed_move.position_fen,
            filename="quiz_position.png",
            flipped=flipped,
        )

        if not board_file:
            logger.error("Failed to generate board image")
            return False

        # Send the quiz
        await interaction.followup.send(embed=embed, file=board_file)

        # Save quiz to database with evaluation data
        quiz = ActiveQuiz(
            channel_id=interaction.channel_id,
            guild_id=interaction.guild_id,
            position_fen=missed_move.position_fen,
            correct_move_san=missed_move.best_move_san,
            played_move_san=missed_move.player_move_san,
            game_url=game.game_url,
            player_username=player.display_name or player.username,
            opponent_username=game.opponent,
            move_number=missed_move.move_number,
            difficulty=difficulty,
            eval_before=missed_move.eval_before,
            eval_after_best=missed_move.eval_after_best,
            eval_after_played=missed_move.eval_after_played,
            started_at=datetime.utcnow(),
        )
        await self.db.save_quiz(quiz)

        return True

    async def check_answer(
        self, channel_id: int, user_move_str: str, user_id: int, user_name: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a user's answer is correct.

        Args:
            channel_id: The channel where the quiz is active
            user_move_str: The user's move input
            user_id: The Discord user ID
            user_name: The user's display name

        Returns:
            Tuple of (is_correct, message). Message is None if no active quiz.
        """
        quiz = await self.db.get_quiz(channel_id)
        if not quiz:
            return False, None

        # Parse the user's move
        board = chess.Board(quiz.position_fen)
        user_move = parse_user_move(user_move_str, board)

        if user_move is None:
            return False, f"'{user_move_str}' is not a valid move in this position."

        # Check if correct by comparing SAN notation
        user_move_san = board.san(user_move)
        if user_move_san == quiz.correct_move_san:
            # Get the user's attempt count (previous wrong attempts + this correct one)
            attempts = json.loads(quiz.attempts or "{}")
            user_attempts = attempts.get(str(user_id), 0) + 1
            await self.end_quiz(
                channel_id, winner_id=user_id, winner_name=user_name, attempts=user_attempts
            )
            return True, None
        else:
            # Track the incorrect attempt
            attempts = json.loads(quiz.attempts or "{}")
            user_key = str(user_id)
            attempts[user_key] = attempts.get(user_key, 0) + 1
            await self.db.update_quiz_attempts(channel_id, json.dumps(attempts))
            return False, "Incorrect, try again!"

    async def reveal_answer(self, channel_id: int) -> Optional[str]:
        """
        Reveal the answer for an active quiz.

        Args:
            channel_id: The channel where the quiz is active

        Returns:
            The correct move, or None if no active quiz
        """
        quiz = await self.db.get_quiz(channel_id)
        if not quiz:
            return None

        await self.end_quiz(channel_id, revealed=True)
        return quiz.correct_move_san

    async def end_quiz(
        self,
        channel_id: int,
        winner_id: Optional[int] = None,
        winner_name: Optional[str] = None,
        revealed: bool = False,
        attempts: int = 1,
    ):
        """
        End a quiz (by correct answer or reveal).

        Args:
            channel_id: The channel where the quiz is active
            winner_id: The Discord user ID of the winner (None if revealed)
            winner_name: The winner's display name
            revealed: True if the answer was revealed via /reveal
            attempts: Number of attempts the winner made (for scoring 1/N)
        """
        quiz = await self.db.get_quiz(channel_id)
        if not quiz:
            return

        # Award points to winner before deleting quiz (score = 1/attempts)
        new_score = None
        points_earned = None
        if winner_id and winner_name:
            points_earned = 1.0 / attempts
            new_score = await self.db.add_quiz_score(
                quiz.guild_id, winner_id, winner_name, points=points_earned
            )

        # Delete quiz from database
        await self.db.delete_quiz(channel_id)

        # Send result message
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return

        # Determine platform from game URL
        platform_name = "Lichess" if "lichess" in quiz.game_url else "Chess.com"

        # Determine player color from position (whose turn it was)
        board = chess.Board(quiz.position_fen)
        player_color = "white" if board.turn == chess.WHITE else "black"

        if winner_id:
            # Build description with attempts and score info
            if attempts == 1:
                attempts_text = "on the first try!"
            else:
                attempts_text = f"in {attempts} tries"

            if points_earned is not None:
                points_text = f"+{points_earned:.2f}" if points_earned < 1 else "+1"
                score_text = f" ({points_text}, Total: {new_score:.2f})"
            else:
                score_text = ""

            embed = discord.Embed(
                title="Correct!",
                description=f"<@{winner_id}> found the best move {attempts_text}{score_text}",
                color=discord.Color.green(),
            )
        else:
            embed = discord.Embed(
                title="Answer Revealed",
                color=discord.Color.orange(),
            )

        # Add Stockfish analysis
        best_move_eval = format_eval_comparison(
            quiz.eval_before, quiz.eval_after_best, player_color
        )
        played_move_eval = format_eval_comparison(
            quiz.eval_before, quiz.eval_after_played, player_color
        )

        embed.add_field(
            name=f"Best Move: {quiz.correct_move_san}",
            value=f"```{best_move_eval}```",
            inline=False,
        )

        embed.add_field(
            name=f"Played Move: {quiz.played_move_san}",
            value=f"```{played_move_eval}```",
            inline=False,
        )

        # Get continuation moves after the best move
        board.push_san(quiz.correct_move_san)
        continuations = await get_continuation_moves_async(board.fen(), num_moves=3)

        if continuations:
            continuation_lines = []
            for cont in continuations:
                eval_str = format_eval(cont.eval_centipawns)
                continuation_lines.append(f"{cont.move_san}: {eval_str}")
            continuation_text = "\n".join(continuation_lines)

            embed.add_field(
                name="Best Continuations",
                value=f"```{continuation_text}```",
                inline=False,
            )

        embed.add_field(
            name="Game",
            value=f"[View on {platform_name}]({quiz.game_url})",
            inline=False,
        )

        await channel.send(embed=embed)

    async def _fetch_pgn(self, player, game) -> Optional[str]:
        """
        Fetch PGN for a game from the appropriate platform.

        Args:
            player: The tracked player
            game: The game to fetch

        Returns:
            PGN string or None if fetch fails
        """
        try:
            if player.platform == "lichess":
                return await self.tracker.lichess_client.get_game_pgn(game.game_id)
            else:
                # Pass played_at to fetch the correct monthly archive
                return await self.tracker.chesscom_client.get_game_pgn(
                    player.username, game.game_id, played_at=game.played_at
                )
        except Exception as e:
            logger.error(f"Error fetching PGN for game {game.game_id}: {e}")
            return None
