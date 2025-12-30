"""Quiz analysis logic for finding missed moves in chess games."""

import asyncio
import io
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Optional

import chess
import chess.pgn

from utils.video import QUIZ_STOCKFISH_DEPTH, STOCKFISH_PATHS

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


def _get_stockfish_path() -> Optional[str]:
    """Find a working Stockfish binary path."""
    try:
        from stockfish import Stockfish
    except ImportError:
        return None

    for candidate in STOCKFISH_PATHS:
        try:
            engine = Stockfish(path=candidate, depth=QUIZ_STOCKFISH_DEPTH)
            engine.send_quit_command()
            return candidate
        except Exception:
            continue
    return None


def _analyze_position(args: tuple) -> tuple:
    """
    Analyze a single position with its own Stockfish instance.

    Args:
        args: (index, fen, player_move_uci, is_white)

    Returns:
        (index, best_move_uci, eval_before, eval_after_best, eval_after_played) or None on error
    """
    idx, fen, player_move_uci, is_white, stockfish_path = args

    try:
        from stockfish import Stockfish
        engine = Stockfish(path=stockfish_path, depth=QUIZ_STOCKFISH_DEPTH)

        # Set up the position
        board = chess.Board(fen)
        engine.set_fen_position(fen)

        # Get best move
        best_move_uci = engine.get_best_move()
        if not best_move_uci:
            engine.send_quit_command()
            return (idx, None, None, None, None)

        best_move = chess.Move.from_uci(best_move_uci)
        player_move = chess.Move.from_uci(player_move_uci)

        # If best move equals player move, no blunder
        if best_move == player_move:
            engine.send_quit_command()
            return (idx, None, None, None, None)

        # Evaluate position before move
        eval_data = engine.get_evaluation()
        if eval_data["type"] == "cp":
            eval_before = float(eval_data["value"])
        elif eval_data["type"] == "mate":
            mate_moves = eval_data["value"]
            eval_before = 10000 - abs(mate_moves) if mate_moves > 0 else -10000 + abs(mate_moves)
        else:
            eval_before = 0.0

        # Adjust for side to move (Stockfish returns from side-to-move perspective)
        if board.turn == chess.BLACK:
            eval_before = -eval_before

        # Evaluate after best move
        board_after_best = board.copy()
        board_after_best.push(best_move)
        engine.set_fen_position(board_after_best.fen())
        eval_data = engine.get_evaluation()
        if eval_data["type"] == "cp":
            eval_after_best = float(eval_data["value"])
        elif eval_data["type"] == "mate":
            mate_moves = eval_data["value"]
            eval_after_best = 10000 - abs(mate_moves) if mate_moves > 0 else -10000 + abs(mate_moves)
        else:
            eval_after_best = 0.0
        if board_after_best.turn == chess.BLACK:
            eval_after_best = -eval_after_best

        # Evaluate after player's move
        board_after_player = board.copy()
        board_after_player.push(player_move)
        engine.set_fen_position(board_after_player.fen())
        eval_data = engine.get_evaluation()
        if eval_data["type"] == "cp":
            eval_after_played = float(eval_data["value"])
        elif eval_data["type"] == "mate":
            mate_moves = eval_data["value"]
            eval_after_played = 10000 - abs(mate_moves) if mate_moves > 0 else -10000 + abs(mate_moves)
        else:
            eval_after_played = 0.0
        if board_after_player.turn == chess.BLACK:
            eval_after_played = -eval_after_played

        engine.send_quit_command()
        return (idx, best_move_uci, eval_before, eval_after_best, eval_after_played)

    except Exception as e:
        logger.error(f"Error analyzing position {idx}: {e}")
        return (idx, None, None, None, None)


def find_missed_moves_parallel(
    pgn: str,
    player_color: str,
    min_eval_loss: float = DEFAULT_MIN_EVAL_LOSS,
) -> list[MissedMove]:
    """
    Analyze a game using parallel Stockfish instances.

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

    stockfish_path = _get_stockfish_path()
    if not stockfish_path:
        logger.warning("Stockfish not available for quiz analysis")
        return []

    # Collect all positions where the player moved
    board = game.board()
    positions_to_analyze = []
    is_white = player_color == "white"

    for move in game.mainline_moves():
        is_white_turn = board.turn == chess.WHITE
        move_number = board.fullmove_number

        if is_white_turn == is_white:
            positions_to_analyze.append({
                "idx": len(positions_to_analyze),
                "fen": board.fen(),
                "player_move": move,
                "player_move_uci": move.uci(),
                "player_move_san": board.san(move),
                "move_number": move_number,
            })

        board.push(move)

    if not positions_to_analyze:
        return []

    logger.info(f"Analyzing {len(positions_to_analyze)} positions in parallel...")

    # Prepare tasks for parallel execution
    tasks = [
        (pos["idx"], pos["fen"], pos["player_move_uci"], is_white, stockfish_path)
        for pos in positions_to_analyze
    ]

    # Run analysis in parallel
    num_workers = min(len(tasks), os.cpu_count() or 4)
    results = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_analyze_position, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results[result[0]] = result

    # Build MissedMove objects from results
    missed_moves = []
    for pos in positions_to_analyze:
        idx = pos["idx"]
        if idx not in results:
            continue

        _, best_move_uci, eval_before, eval_after_best, eval_after_played = results[idx]

        if best_move_uci is None:
            continue

        best_move = chess.Move.from_uci(best_move_uci)

        # Calculate loss from player's perspective
        if is_white:
            eval_loss = eval_after_best - eval_after_played
        else:
            eval_loss = eval_after_played - eval_after_best

        if eval_loss >= min_eval_loss:
            missed_moves.append(
                MissedMove(
                    move_number=pos["move_number"],
                    player_color=player_color,
                    position_fen=pos["fen"],
                    player_move=pos["player_move"],
                    player_move_san=pos["player_move_san"],
                    best_move=best_move,
                    best_move_san=chess.Board(pos["fen"]).san(best_move),
                    eval_loss=eval_loss,
                    eval_before=eval_before,
                    eval_after_best=eval_after_best,
                    eval_after_played=eval_after_played,
                )
            )

    logger.info(f"Found {len(missed_moves)} blunders (>= {min_eval_loss} cp)")
    return missed_moves


async def find_missed_moves_async(
    pgn: str,
    player_color: str,
    min_eval_loss: float = DEFAULT_MIN_EVAL_LOSS,
) -> list[MissedMove]:
    """
    Async wrapper for find_missed_moves_parallel that runs in a thread pool.

    This prevents blocking the Discord event loop during Stockfish analysis.
    """
    loop = asyncio.get_event_loop()
    func = partial(find_missed_moves_parallel, pgn, player_color, min_eval_loss)
    return await loop.run_in_executor(None, func)


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
