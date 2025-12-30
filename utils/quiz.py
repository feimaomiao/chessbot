"""Quiz analysis logic for finding missed moves in chess games."""

import asyncio
import io
import json
import logging
import os
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import chess
import chess.pgn

from utils.video import QUIZ_STOCKFISH_DEPTH, STOCKFISH_PATHS

logger = logging.getLogger(__name__)

# Default minimum evaluation loss in centipawns to consider a move as a "blunder"
# 300+ centipawns is typically considered a blunder in chess
DEFAULT_MIN_EVAL_LOSS = 300

# Maximum evaluation deficit before the blunder to include in quizzes
# If player was already losing by more than this, skip the position
# (no point quizzing on a blunder when already down 5+ pawns)
DEFAULT_MAX_LOSING_EVAL = 500

# Quiz evaluation cache settings
QUIZ_CACHE_SIZE = 5000
QUIZ_CACHE_FILE = "./data/quiz_eval_cache.json"


class QuizEvalCache:
    """Thread-safe LRU cache for quiz position evaluations."""

    def __init__(self, maxsize: int = QUIZ_CACHE_SIZE):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._file_path: Optional[str] = None

    def get(self, fen: str) -> Optional[float]:
        """Get cached evaluation for a FEN position."""
        with self._lock:
            if fen in self._cache:
                self._cache.move_to_end(fen)
                self._hits += 1
                return self._cache[fen]
            self._misses += 1
            return None

    def set(self, fen: str, score: float) -> None:
        """Cache an evaluation for a FEN position."""
        with self._lock:
            if fen in self._cache:
                self._cache.move_to_end(fen)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[fen] = score

    def get_stats(self) -> tuple[int, int]:
        """Get cache hit and miss counts."""
        with self._lock:
            return self._hits, self._misses

    def reset_stats(self) -> tuple[int, int]:
        """Reset and return cache hit/miss counts."""
        with self._lock:
            hits, misses = self._hits, self._misses
            self._hits = 0
            self._misses = 0
            return hits, misses

    def save_to_file(self, file_path: Optional[str] = None) -> bool:
        """Save cache to a JSON file."""
        file_path = file_path or self._file_path
        if not file_path:
            return False
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {"cache": dict(self._cache)}
            with open(file_path, "w") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save quiz cache: {e}")
            return False

    def load_from_file(self, file_path: str) -> bool:
        """Load cache from a JSON file."""
        self._file_path = file_path
        try:
            if not Path(file_path).exists():
                return False
            with open(file_path, "r") as f:
                data = json.load(f)
            with self._lock:
                self._cache = OrderedDict(data.get("cache", {}))
            logger.info(f"Loaded {len(self._cache)} quiz cache entries")
            return True
        except Exception as e:
            logger.error(f"Failed to load quiz cache: {e}")
            return False


# Global quiz evaluation cache
_quiz_eval_cache: Optional[QuizEvalCache] = None


def get_quiz_eval_cache() -> QuizEvalCache:
    """Get or create the global quiz evaluation cache."""
    global _quiz_eval_cache
    if _quiz_eval_cache is None:
        _quiz_eval_cache = QuizEvalCache()
        _quiz_eval_cache.load_from_file(QUIZ_CACHE_FILE)
    return _quiz_eval_cache


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


def _evaluate_fen(engine, fen: str, board_turn: chess.Color, cache: QuizEvalCache) -> float:
    """
    Evaluate a position, using cache if available.

    Args:
        engine: Stockfish engine instance
        fen: FEN string of position to evaluate
        board_turn: Whose turn it is in the position
        cache: Quiz evaluation cache

    Returns:
        Evaluation in centipawns from white's perspective
    """
    # Check cache first
    cached = cache.get(fen)
    if cached is not None:
        return cached

    # Evaluate with engine
    engine.set_fen_position(fen)
    eval_data = engine.get_evaluation()

    if eval_data["type"] == "cp":
        eval_score = float(eval_data["value"])
    elif eval_data["type"] == "mate":
        mate_moves = eval_data["value"]
        eval_score = 10000 - abs(mate_moves) if mate_moves > 0 else -10000 + abs(mate_moves)
    else:
        eval_score = 0.0

    # Adjust for side to move (Stockfish returns from side-to-move perspective)
    if board_turn == chess.BLACK:
        eval_score = -eval_score

    # Cache the result
    cache.set(fen, eval_score)
    return eval_score


def _analyze_position(args: tuple) -> tuple:
    """
    Analyze a single position with its own Stockfish instance.

    Args:
        args: (index, fen, player_move_uci, is_white, stockfish_path)

    Returns:
        (index, best_move_uci, eval_before, eval_after_best, eval_after_played) or None on error
    """
    idx, fen, player_move_uci, is_white, stockfish_path = args
    cache = get_quiz_eval_cache()

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

        # Evaluate position before move (with caching)
        eval_before = _evaluate_fen(engine, fen, board.turn, cache)

        # Evaluate after best move (with caching)
        board_after_best = board.copy()
        board_after_best.push(best_move)
        eval_after_best = _evaluate_fen(
            engine, board_after_best.fen(), board_after_best.turn, cache
        )

        # Evaluate after player's move (with caching)
        board_after_player = board.copy()
        board_after_player.push(player_move)
        eval_after_played = _evaluate_fen(
            engine, board_after_player.fen(), board_after_player.turn, cache
        )

        engine.send_quit_command()
        return (idx, best_move_uci, eval_before, eval_after_best, eval_after_played)

    except Exception as e:
        logger.error(f"Error analyzing position {idx}: {e}")
        return (idx, None, None, None, None)


def find_missed_moves_parallel(
    pgn: str,
    player_color: str,
    min_eval_loss: float = DEFAULT_MIN_EVAL_LOSS,
    max_losing_eval: float = DEFAULT_MAX_LOSING_EVAL,
) -> list[MissedMove]:
    """
    Analyze a game using parallel Stockfish instances.

    Args:
        pgn: PGN string of the game
        player_color: "white" or "black" - which player to analyze
        min_eval_loss: Minimum centipawn loss to consider a move as "missed"
        max_losing_eval: Skip positions where player was already losing by more than this

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

    # Get cache and reset stats for this operation
    cache = get_quiz_eval_cache()
    cache.reset_stats()

    # Run analysis in parallel with timing
    num_workers = min(len(tasks), os.cpu_count() or 4)
    results = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_analyze_position, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results[result[0]] = result

    elapsed_ms = (time.time() - start_time) * 1000

    # Record stats and save cache
    cache_hits, cache_misses = cache.get_stats()
    total_evals = cache_hits + cache_misses

    from utils.stats import get_stats_tracker
    get_stats_tracker().record_quiz_evaluation(total_evals, elapsed_ms, cache_hits)
    cache.save_to_file(QUIZ_CACHE_FILE)

    logger.info(
        f"Quiz analysis completed in {elapsed_ms:.0f}ms "
        f"({total_evals} evals, {cache_hits} cache hits)"
    )

    # Build MissedMove objects from results
    missed_moves = []
    for pos in positions_to_analyze:
        idx = pos["idx"]
        if idx not in results:
            continue

        _, best_move_uci, eval_before, eval_after_best, eval_after_played = results[idx]

        if best_move_uci is None:
            continue

        # Skip positions where player was already losing badly
        # (no point quizzing on a blunder when already in a hopeless position)
        if is_white and eval_before < -max_losing_eval:
            continue
        if not is_white and eval_before > max_losing_eval:
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
    max_losing_eval: float = DEFAULT_MAX_LOSING_EVAL,
) -> list[MissedMove]:
    """
    Async wrapper for find_missed_moves_parallel that runs in a thread pool.

    This prevents blocking the Discord event loop during Stockfish analysis.
    """
    loop = asyncio.get_event_loop()
    func = partial(
        find_missed_moves_parallel, pgn, player_color, min_eval_loss, max_losing_eval
    )
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
