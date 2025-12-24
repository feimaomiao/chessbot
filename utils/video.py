"""Video generation for chess game replays with evaluation bar."""

import io
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import chess.svg
import cairosvg
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Piece values for material-based evaluation (fallback)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Video settings
FRAME_DURATION_MS = 1000  # 1 second per move
BOARD_SIZE = 400
EVAL_BAR_WIDTH = 30
EVAL_BAR_HEIGHT = BOARD_SIZE

# Parallelization settings
MAX_WORKERS = None  # None = use cpu_count() * 5 for ThreadPoolExecutor

# Cache settings
EVAL_CACHE_SIZE = 10000  # Number of positions to cache

# Stockfish settings
STOCKFISH_DEPTH = 12  # Analysis depth (higher = slower but more accurate)
STOCKFISH_PATHS = [
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "stockfish",  # Try PATH
]


class EvalCache:
    """Thread-safe LRU cache for position evaluations."""

    def __init__(self, maxsize: int = EVAL_CACHE_SIZE):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, fen: str) -> Optional[float]:
        """Get cached evaluation for a FEN position."""
        with self._lock:
            if fen in self._cache:
                # Move to end (most recently used)
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
                    # Remove oldest (least recently used)
                    self._cache.popitem(last=False)
                self._cache[fen] = score

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": f"{hit_rate:.1f}%",
            }

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# Global evaluation cache
_eval_cache = EvalCache()


def get_eval_cache() -> EvalCache:
    """Get the global evaluation cache."""
    return _eval_cache


class StockfishEvaluator:
    """Wrapper for Stockfish engine evaluation."""

    def __init__(self, path: Optional[str] = None, depth: int = STOCKFISH_DEPTH):
        self.engine = None
        self.depth = depth
        self._init_engine(path)

    def _init_engine(self, path: Optional[str] = None):
        """Initialize the Stockfish engine."""
        try:
            from stockfish import Stockfish
        except ImportError:
            logger.warning("stockfish package not installed")
            return

        # Find Stockfish binary
        stockfish_path = path or os.environ.get("STOCKFISH_PATH")

        if not stockfish_path:
            for candidate in STOCKFISH_PATHS:
                try:
                    self.engine = Stockfish(path=candidate, depth=self.depth)
                    logger.info(f"Stockfish initialized at: {candidate}")
                    return
                except Exception:
                    continue

        if stockfish_path:
            try:
                self.engine = Stockfish(path=stockfish_path, depth=self.depth)
                logger.info(f"Stockfish initialized at: {stockfish_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize Stockfish at {stockfish_path}: {e}")

        if not self.engine:
            logger.warning(
                "Stockfish binary not found. Set STOCKFISH_PATH environment variable "
                "or install Stockfish: brew install stockfish (macOS) / "
                "apt install stockfish (Linux)"
            )

    @property
    def available(self) -> bool:
        """Check if Stockfish is available."""
        return self.engine is not None

    def evaluate(self, board: chess.Board, use_cache: bool = True) -> float:
        """
        Evaluate a position using Stockfish.

        Args:
            board: Chess board position
            use_cache: Whether to use the evaluation cache

        Returns:
            Evaluation in centipawns from white's perspective.
            Returns ±10000 for mate positions.
        """
        if not self.engine:
            return calculate_material_eval(board)

        fen = board.fen()

        # Check cache first
        if use_cache:
            cache = get_eval_cache()
            cached = cache.get(fen)
            if cached is not None:
                return cached

        try:
            self.engine.set_fen_position(fen)
            evaluation = self.engine.get_evaluation()

            if evaluation["type"] == "cp":
                result = float(evaluation["value"])
            elif evaluation["type"] == "mate":
                # Mate in N moves - return large value
                mate_moves = evaluation["value"]
                if mate_moves > 0:
                    result = 10000 - mate_moves  # White winning
                else:
                    result = -10000 - mate_moves  # Black winning
            else:
                result = 0.0

            # Store in cache
            if use_cache:
                cache.set(fen, result)

            return result

        except Exception as e:
            logger.error(f"Stockfish evaluation error: {e}")
            return calculate_material_eval(board)

    def evaluate_positions(
        self, positions: list[chess.Board], parallel: bool = True
    ) -> list[float]:
        """
        Evaluate multiple positions.

        Args:
            positions: List of chess board positions
            parallel: Whether to use parallel evaluation (requires multiple Stockfish instances)

        Returns:
            List of evaluations in centipawns
        """
        if not parallel or not self.available:
            return [self.evaluate(board) for board in positions]

        return self._evaluate_parallel(positions)

    def _evaluate_parallel(self, positions: list[chess.Board]) -> list[float]:
        """
        Evaluate positions in parallel using multiple Stockfish instances.
        Uses cache to avoid re-evaluating known positions.

        Args:
            positions: List of chess board positions

        Returns:
            List of evaluations in centipawns (in order)
        """
        cache = get_eval_cache()
        results = [0.0] * len(positions)

        # First pass: check cache and collect uncached positions
        uncached_tasks = []
        for i, board in enumerate(positions):
            fen = board.fen()
            cached = cache.get(fen)
            if cached is not None:
                results[i] = cached
            else:
                uncached_tasks.append((i, board, fen))

        # If all cached, return early
        if not uncached_tasks:
            logger.info(f"All {len(positions)} positions found in cache")
            return results

        logger.info(
            f"Cache: {len(positions) - len(uncached_tasks)}/{len(positions)} hits, "
            f"evaluating {len(uncached_tasks)} positions"
        )

        try:
            from stockfish import Stockfish
        except ImportError:
            # Fall back to sequential evaluation for uncached
            for idx, board, fen in uncached_tasks:
                result = calculate_material_eval(board)
                results[idx] = result
                cache.set(fen, result)
            return results

        # Find the stockfish path that works
        stockfish_path = None
        for candidate in STOCKFISH_PATHS:
            try:
                test_engine = Stockfish(path=candidate, depth=self.depth)
                stockfish_path = candidate
                del test_engine
                break
            except Exception:
                continue

        if not stockfish_path:
            # Fall back to sequential evaluation for uncached
            for idx, board, fen in uncached_tasks:
                result = calculate_material_eval(board)
                results[idx] = result
                cache.set(fen, result)
            return results

        def evaluate_single(args: tuple[int, chess.Board, str]) -> tuple[int, float, str]:
            """Evaluate a single position with its own Stockfish instance."""
            idx, board, fen = args
            try:
                engine = Stockfish(path=stockfish_path, depth=self.depth)
                engine.set_fen_position(fen)
                evaluation = engine.get_evaluation()

                if evaluation["type"] == "cp":
                    result = float(evaluation["value"])
                elif evaluation["type"] == "mate":
                    mate_moves = evaluation["value"]
                    if mate_moves > 0:
                        result = 10000 - mate_moves
                    else:
                        result = -10000 - mate_moves
                else:
                    result = 0.0

                del engine
                return (idx, result, fen)
            except Exception as e:
                logger.debug(f"Parallel eval error for position {idx}: {e}")
                return (idx, calculate_material_eval(board), fen)

        # Use ThreadPoolExecutor for parallel evaluation
        # Limit workers to avoid spawning too many Stockfish processes
        num_workers = min(len(uncached_tasks), os.cpu_count() or 4)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(evaluate_single, task): task[0]
                for task in uncached_tasks
            }

            for future in as_completed(futures):
                idx, score, fen = future.result()
                results[idx] = score
                # Store in cache for future use
                cache.set(fen, score)

        return results

    def close(self):
        """Clean up the Stockfish engine."""
        if self.engine:
            try:
                del self.engine
            except Exception as e:
                logger.debug(f"Error closing Stockfish engine: {e}")
            self.engine = None


# Global evaluator instance (lazy initialization)
_stockfish_evaluator: Optional[StockfishEvaluator] = None


def get_stockfish_evaluator() -> StockfishEvaluator:
    """Get or create the global Stockfish evaluator."""
    global _stockfish_evaluator
    if _stockfish_evaluator is None:
        _stockfish_evaluator = StockfishEvaluator()
    return _stockfish_evaluator


def calculate_material_eval(board: chess.Board) -> float:
    """
    Calculate a simple material-based evaluation.

    Returns:
        Evaluation in centipawns from white's perspective.
        Positive = white advantage, negative = black advantage.
    """
    eval_score = 0

    for piece_type in PIECE_VALUES:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        eval_score += PIECE_VALUES[piece_type] * (white_pieces - black_pieces)

    return eval_score


def eval_to_win_probability(centipawns: float) -> float:
    """
    Convert centipawn evaluation to win probability (0-1).
    Uses a sigmoid-like function.
    """
    # Clamp to reasonable range
    centipawns = max(-1500, min(1500, centipawns))
    # Sigmoid transformation
    return 1 / (1 + 10 ** (-centipawns / 400))


def render_board_image(
    board: chess.Board,
    size: int = BOARD_SIZE,
    last_move: Optional[chess.Move] = None,
) -> Image.Image:
    """
    Render a chess board to a PIL Image.

    Args:
        board: Chess board position
        size: Size of the output image in pixels
        last_move: Optional move to highlight (from/to squares)

    Returns:
        PIL Image of the board
    """
    svg_data = chess.svg.board(
        board,
        size=size,
        lastmove=last_move,
        coordinates=True,
        colors={
            "square light": "#f0d9b5",
            "square dark": "#b58863",
            "margin": "#212121",
            "coord": "#e0e0e0",
        },
    )

    png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
    return Image.open(io.BytesIO(png_data))


def format_eval_text(centipawns: float) -> str:
    """
    Format evaluation for display.

    Args:
        centipawns: Evaluation in centipawns

    Returns:
        Formatted string (e.g., "+1.5", "-0.3", "M3", "M-2")
    """
    if centipawns >= 9900:
        # Mate for white
        mate_in = 10000 - int(centipawns)
        return f"M{mate_in}" if mate_in > 0 else "M"
    elif centipawns <= -9900:
        # Mate for black
        mate_in = 10000 + int(centipawns)
        return f"M{mate_in}" if mate_in > 0 else "M"
    else:
        # Regular evaluation in pawns
        pawns = centipawns / 100
        if pawns >= 0:
            return f"+{pawns:.1f}"
        else:
            return f"{pawns:.1f}"


def render_eval_bar(
    eval_score: float,
    width: int = EVAL_BAR_WIDTH,
    height: int = EVAL_BAR_HEIGHT,
) -> Image.Image:
    """
    Render an evaluation bar with numerical display.

    Args:
        eval_score: Evaluation in centipawns (positive = white advantage)
        width: Width of the bar
        height: Height of the bar

    Returns:
        PIL Image of the evaluation bar
    """
    img = Image.new("RGB", (width, height), color="#404040")
    draw = ImageDraw.Draw(img)

    # Convert to win probability
    win_prob = eval_to_win_probability(eval_score)

    # Calculate where white section ends (from top)
    # win_prob=1 means all white (white winning), win_prob=0 means all black
    white_height = int(height * (1 - win_prob))

    # Draw black section (top)
    draw.rectangle([0, 0, width, white_height], fill="#1a1a1a")

    # Draw white section (bottom)
    draw.rectangle([0, white_height, width, height], fill="#f0f0f0")

    # Draw center line
    center_y = height // 2
    draw.line([(0, center_y), (width, center_y)], fill="#808080", width=1)

    # Draw evaluation text
    eval_text = format_eval_text(eval_score)

    # Position text in the larger section for visibility
    # Use contrasting color based on background
    if win_prob > 0.5:
        # More white (bottom), draw in white section
        text_y = height - 15
        text_color = "#1a1a1a"
    else:
        # More black (top), draw in black section
        text_y = 5
        text_color = "#f0f0f0"

    # Center text horizontally
    bbox = draw.textbbox((0, 0), eval_text)
    text_width = bbox[2] - bbox[0]
    text_x = (width - text_width) // 2

    draw.text((text_x, text_y), eval_text, fill=text_color)

    return img


def render_frame(
    board: chess.Board,
    eval_score: Optional[float] = None,
    board_size: int = BOARD_SIZE,
    last_move: Optional[chess.Move] = None,
) -> Image.Image:
    """
    Render a single frame with board and evaluation bar.

    Args:
        board: Chess board position
        eval_score: Optional evaluation in centipawns. If None, calculated from material.
        board_size: Size of the board in pixels
        last_move: Optional move to highlight on the board

    Returns:
        PIL Image combining board and eval bar
    """
    if eval_score is None:
        eval_score = calculate_material_eval(board)

    # Render components
    board_img = render_board_image(board, board_size, last_move)
    eval_bar = render_eval_bar(eval_score, EVAL_BAR_WIDTH, board_size)

    # Combine: eval bar on the left, board on the right
    total_width = EVAL_BAR_WIDTH + board_size
    combined = Image.new("RGB", (total_width, board_size), color="#212121")
    combined.paste(eval_bar, (0, 0))
    combined.paste(board_img, (EVAL_BAR_WIDTH, 0))

    return combined


def _render_frame_task(
    args: tuple[int, chess.Board, Optional[float], int, Optional[chess.Move]]
) -> tuple[int, Image.Image]:
    """
    Render a single frame (for parallel processing).

    Args:
        args: Tuple of (index, board, eval_score, board_size, last_move)

    Returns:
        Tuple of (index, rendered_frame) to maintain order
    """
    idx, board, eval_score, board_size, last_move = args
    frame = render_frame(board, eval_score, board_size, last_move=last_move)
    return (idx, frame)


def render_frames_parallel(
    positions: list[tuple[chess.Board, chess.Move | None]],
    evaluations: Optional[list[float]],
    board_size: int = BOARD_SIZE,
    max_workers: Optional[int] = MAX_WORKERS,
) -> list[Image.Image]:
    """
    Render all frames in parallel while maintaining order.

    Args:
        positions: List of (board, move) tuples
        evaluations: Optional list of evaluation scores
        board_size: Size of the board in pixels
        max_workers: Maximum number of worker threads (None = default)

    Returns:
        List of rendered frames in order
    """
    # Prepare tasks with indices to maintain order
    tasks = []
    for i, (board, move) in enumerate(positions):
        eval_score = evaluations[i] if evaluations and i < len(evaluations) else None
        tasks.append((i, board, eval_score, board_size, move))

    # Render frames in parallel
    frames = [None] * len(positions)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_render_frame_task, task): task[0] for task in tasks}

        for future in as_completed(futures):
            idx, frame = future.result()
            frames[idx] = frame

    return frames


def parse_pgn_positions(pgn_str: str) -> list[tuple[chess.Board, chess.Move | None]]:
    """
    Parse a PGN string and return all positions with the move that led to them.

    Returns:
        List of (board, move) tuples. First entry has move=None (starting position).
    """
    if not pgn_str:
        return []

    try:
        pgn = chess.pgn.read_game(io.StringIO(pgn_str))
        if pgn is None:
            return []

        positions = []
        board = pgn.board()

        # Add starting position
        positions.append((board.copy(), None))

        # Add each position after a move
        for move in pgn.mainline_moves():
            board.push(move)
            positions.append((board.copy(), move))

        return positions

    except Exception as e:
        logger.error(f"Error parsing PGN: {e}")
        return []


def generate_game_video(
    pgn_str: str,
    evaluations: Optional[list[float]] = None,
    frame_duration_ms: int = FRAME_DURATION_MS,
    board_size: int = BOARD_SIZE,
    use_stockfish: bool = True,
) -> Optional[bytes]:
    """
    Generate a video of a chess game from PGN.

    Args:
        pgn_str: PGN string of the game
        evaluations: Optional list of centipawn evaluations per position.
                    If None and use_stockfish=True, Stockfish is used.
                    If None and use_stockfish=False, material evaluation is used.
        frame_duration_ms: Duration of each frame in milliseconds
        board_size: Size of the board in pixels
        use_stockfish: Whether to use Stockfish for evaluation (default True)

    Returns:
        MP4 video as bytes, or None if generation fails
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        logger.error("imageio not installed. Run: pip install imageio[ffmpeg]")
        return None

    positions = parse_pgn_positions(pgn_str)
    if not positions:
        logger.error("No positions found in PGN")
        return None

    # Calculate evaluations if not provided
    if evaluations is None and use_stockfish:
        evaluator = get_stockfish_evaluator()
        if evaluator.available:
            logger.info(f"Evaluating {len(positions)} positions with Stockfish...")
            boards = [board for board, _ in positions]
            evaluations = evaluator.evaluate_positions(boards)
            logger.info("Stockfish evaluation complete")
        else:
            logger.info("Stockfish not available, using material evaluation")

    # Generate frames in parallel
    logger.info(f"Rendering {len(positions)} frames in parallel...")
    frames = render_frames_parallel(positions, evaluations, board_size)
    logger.info("Frame rendering complete")

    # Calculate FPS from frame duration
    fps = 1000 / frame_duration_ms

    # Write video to temporary file, then read bytes
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        # Convert PIL images to numpy arrays for imageio
        import numpy as np
        frame_arrays = [np.array(f) for f in frames]

        # Write video
        iio.imwrite(
            tmp_path,
            frame_arrays,
            fps=fps,
            codec="libx264",
            plugin="pyav",
        )

        # Read the video bytes
        video_bytes = tmp_path.read_bytes()
        return video_bytes

    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return None

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


async def generate_game_video_async(
    pgn_str: str,
    evaluations: Optional[list[float]] = None,
    frame_duration_ms: int = FRAME_DURATION_MS,
    board_size: int = BOARD_SIZE,
    use_stockfish: bool = True,
) -> Optional[bytes]:
    """
    Async wrapper for generate_game_video.
    Runs the CPU-intensive video generation in a thread pool.

    Args:
        pgn_str: PGN string of the game
        evaluations: Optional list of centipawn evaluations per position
        frame_duration_ms: Duration of each frame in milliseconds
        board_size: Size of the board in pixels
        use_stockfish: Whether to use Stockfish for evaluation (default True)

    Returns:
        MP4 video as bytes, or None if generation fails
    """
    import asyncio
    from functools import partial

    loop = asyncio.get_event_loop()
    func = partial(
        generate_game_video,
        pgn_str,
        evaluations,
        frame_duration_ms,
        board_size,
        use_stockfish,
    )
    return await loop.run_in_executor(None, func)
