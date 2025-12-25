"""
Accuracy calculation for chess games using Lichess formula.

Based on: https://lichess.org/page/accuracy
"""

import logging
import math
from typing import Optional

import chess

from utils.video import get_stockfish_evaluator, parse_pgn_positions

logger = logging.getLogger(__name__)


def centipawns_to_win_pct(cp: float) -> float:
    """
    Convert centipawns to win percentage using Lichess formula.

    Args:
        cp: Centipawn evaluation from white's perspective

    Returns:
        Win percentage for white (0-100)
    """
    # Clamp extreme values
    cp = max(-10000, min(10000, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(win_pct_before: float, win_pct_after: float) -> float:
    """
    Calculate accuracy for a single move using Lichess formula.

    Args:
        win_pct_before: Win percentage before the move (from player's perspective)
        win_pct_after: Win percentage after the move (from player's perspective)

    Returns:
        Accuracy percentage for this move (0-100)
    """
    # Calculate how much winning chances decreased
    loss = max(0, win_pct_before - win_pct_after)

    # Lichess formula
    acc = 103.1668 * math.exp(-0.04354 * loss) - 3.1669

    return max(0, min(100, acc))


def calculate_game_accuracy(
    evaluations: list[float],
    player_color: str,
) -> Optional[float]:
    """
    Calculate overall game accuracy for a player.

    Args:
        evaluations: List of centipawn evaluations for each position (from white's perspective)
        player_color: "white" or "black"

    Returns:
        Game accuracy percentage (0-100), or None if not enough data
    """
    if len(evaluations) < 2:
        return None

    is_white = player_color == "white"
    move_accuracies = []

    # Iterate through moves (every other position belongs to this player)
    # Position 0 is starting position, position 1 is after white's first move, etc.
    for i in range(1, len(evaluations)):
        # Determine if this move was made by the player
        # Odd indices (1, 3, 5...) are after white's moves
        # Even indices (2, 4, 6...) are after black's moves
        is_white_move = (i % 2 == 1)

        if is_white_move != is_white:
            continue  # Skip opponent's moves

        # Get evaluations from player's perspective
        cp_before = evaluations[i - 1]
        cp_after = evaluations[i]

        # Convert to win percentage from player's perspective
        if is_white:
            win_before = centipawns_to_win_pct(cp_before)
            win_after = centipawns_to_win_pct(cp_after)
        else:
            # For black, invert the perspective
            win_before = 100 - centipawns_to_win_pct(cp_before)
            win_after = 100 - centipawns_to_win_pct(cp_after)

        acc = move_accuracy(win_before, win_after)
        move_accuracies.append(acc)

    if not move_accuracies:
        return None

    # Use harmonic mean (like Lichess) to penalize bad moves more
    try:
        harmonic_mean = len(move_accuracies) / sum(1 / max(a, 0.01) for a in move_accuracies)
        return round(harmonic_mean, 1)
    except ZeroDivisionError:
        return None


async def calculate_accuracy_from_pgn(
    pgn: str,
    player_color: str,
) -> Optional[float]:
    """
    Calculate accuracy for a game from its PGN.

    Args:
        pgn: PGN string of the game
        player_color: "white" or "black"

    Returns:
        Accuracy percentage (0-100), or None if calculation fails
    """
    if not pgn:
        return None

    # Parse positions from PGN
    positions = parse_pgn_positions(pgn)
    if len(positions) < 3:  # Need at least a few moves
        return None

    boards = [board for board, _ in positions]

    # Get evaluations using Stockfish
    evaluator = get_stockfish_evaluator()
    if not evaluator.available:
        logger.warning("Stockfish not available for accuracy calculation")
        return None

    try:
        # Evaluate all positions (don't track stats for accuracy calculation)
        evaluations = evaluator.evaluate_positions(boards, parallel=True, track_stats=False)

        # Calculate accuracy
        return calculate_game_accuracy(evaluations, player_color)

    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        return None
