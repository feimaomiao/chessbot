"""Quiz analysis logic for finding missed moves in chess games."""

import io
import logging
import random
from dataclasses import dataclass
from typing import Optional

import chess
import chess.pgn

from utils.video import get_quiz_stockfish_evaluator

logger = logging.getLogger(__name__)

# Default minimum evaluation loss in centipawns to consider a move as a "blunder"
# 300+ centipawns is typically considered a blunder in chess
DEFAULT_MIN_EVAL_LOSS = 300


@dataclass
class MissedMove:
    """Represents a position where the player missed a better move."""

    move_number: int
    player_color: str  # "white" or "black"
    position_fen: str  # FEN before the missed move
    player_move: chess.Move  # The move the player actually made
    player_move_san: str  # SAN notation of player's move
    best_move: chess.Move  # The engine's recommended move
    best_move_san: str  # SAN notation of best move
    eval_loss: float  # Centipawn loss (how bad was the move)
    eval_before: float  # Evaluation before the move
    eval_after_best: float  # Evaluation after best move
    eval_after_played: float  # Evaluation after the blunder


def find_missed_moves(
    pgn: str,
    player_color: str,
    min_eval_loss: float = DEFAULT_MIN_EVAL_LOSS,
) -> list[MissedMove]:
    """
    Analyze a game and find positions where the player missed better moves.

    Args:
        pgn: PGN string of the game
        player_color: "white" or "black" - which player to analyze
        min_eval_loss: Minimum centipawn loss to consider a move as "missed"

    Returns:
        List of MissedMove objects representing positions where the player
        could have played a significantly better move.
    """
    if not pgn:
        return []

    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
        if not game:
            logger.warning("Could not parse PGN")
            return []
    except Exception as e:
        logger.error(f"Error parsing PGN: {e}")
        return []

    evaluator = get_quiz_stockfish_evaluator()
    if not evaluator.available:
        logger.warning("Stockfish not available for quiz analysis")
        return []

    board = game.board()
    missed_moves = []
    is_white = player_color == "white"

    for move in game.mainline_moves():
        is_white_turn = board.turn == chess.WHITE
        move_number = board.fullmove_number

        # Only analyze the tracked player's moves
        if is_white_turn == is_white:
            # Get best move for this position
            best_move = evaluator.get_best_move(board)

            if best_move and best_move != move:
                # Evaluate the position before the move
                position_fen = board.fen()
                eval_before = evaluator.evaluate(board)

                # Evaluate position after player's actual move
                board_after_player = board.copy()
                board_after_player.push(move)
                eval_after_player = evaluator.evaluate(board_after_player)

                # Evaluate position after best move
                board_after_best = board.copy()
                board_after_best.push(best_move)
                eval_after_best = evaluator.evaluate(board_after_best)

                # Calculate loss from player's perspective
                # Positive eval is good for white, negative for black
                if is_white:
                    # White wants higher eval
                    eval_loss = eval_after_best - eval_after_player
                else:
                    # Black wants lower eval (more negative)
                    eval_loss = eval_after_player - eval_after_best

                if eval_loss >= min_eval_loss:
                    missed_moves.append(
                        MissedMove(
                            move_number=move_number,
                            player_color=player_color,
                            position_fen=position_fen,
                            player_move=move,
                            player_move_san=board.san(move),
                            best_move=best_move,
                            best_move_san=board.san(best_move),
                            eval_loss=eval_loss,
                            eval_before=eval_before,
                            eval_after_best=eval_after_best,
                            eval_after_played=eval_after_player,
                        )
                    )

        board.push(move)

    return missed_moves


def select_quiz_position(missed_moves: list[MissedMove]) -> Optional[MissedMove]:
    """
    Select a quiz position from the list of missed moves.

    Args:
        missed_moves: List of missed moves to choose from

    Returns:
        A randomly selected MissedMove, or None if list is empty
    """
    if not missed_moves:
        return None

    return random.choice(missed_moves)


def classify_difficulty(eval_loss: float) -> str:
    """
    Classify a blunder's difficulty based on evaluation loss.

    The difficulty refers to how hard it is to find the correct move,
    not how bad the blunder was.

    Args:
        eval_loss: The centipawn loss of the blunder

    Returns:
        Difficulty classification: "easy", "medium", or "hard"
    """
    if eval_loss >= 600:
        return "easy"  # Very obvious blunder, easy to find the right move
    elif eval_loss >= 400:
        return "medium"  # Clear blunder
    else:
        return "hard"  # Trickier to spot the best move
