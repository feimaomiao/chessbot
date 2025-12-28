from typing import Optional

import chess

from config import TIME_CONTROL_THRESHOLDS

def parse_time_control(time_control_str: str) -> tuple[str, str]:
    """
    Parse time control string and return (category, display_string).

    Handles formats like:
    - "180" (chess.com seconds)
    - "180+2" (base + increment)
    - "bullet", "blitz", etc. (lichess categories)
    """
    if not time_control_str:
        return "unknown", "Unknown"

    # If it's already a category name
    if time_control_str.lower() in TIME_CONTROL_THRESHOLDS:
        return time_control_str.lower(), time_control_str.capitalize()

    # Parse numeric time controls
    try:
        if "+" in time_control_str:
            base, increment = time_control_str.split("+")
            base_seconds = int(base)
            increment_seconds = int(increment)
            display = f"{base_seconds // 60}+{increment_seconds}"
        else:
            base_seconds = int(time_control_str)
            increment_seconds = 0
            display = f"{base_seconds // 60} min"

        # Estimate total time (base + 40 moves worth of increment)
        estimated_total = base_seconds + (increment_seconds * 40)

        for category, (min_time, max_time) in TIME_CONTROL_THRESHOLDS.items():
            if min_time <= estimated_total < max_time:
                return category, display

        return "classical", display
    except (ValueError, AttributeError):
        return "unknown", time_control_str


def format_rating_change(change: int) -> str:
    """Format rating change with + or - prefix."""
    if change > 0:
        return f"+{change}"
    return str(change)


def get_result_emoji(result: str) -> str:
    """Get emoji for game result."""
    emoji_map = {
        "win": "🏆",
        "loss": "❌",
        "draw": "🤝",
    }
    return emoji_map.get(result.lower(), "❓")


def get_time_control_emoji(category: str) -> str:
    """Get emoji for time control category."""
    emoji_map = {
        "bullet": "🚀",
        "blitz": "⚡",
        "rapid": "🕐",
        "classical": "🏛️",
        "unknown": "🎮",
    }
    return emoji_map.get(category.lower(), "🎮")


def format_platform_name(platform: str) -> str:
    """Format platform name for display."""
    return {
        "chesscom": "Chess.com",
        "lichess": "Lichess",
    }.get(platform, platform)


def infer_termination_from_fen(fen: str, result: str) -> Optional[str]:
    """
    Infer game termination from final FEN position and result.

    Can only detect checkmate and stalemate from the board position.
    Other terminations (timeout, resign, agreed) cannot be inferred.

    Args:
        fen: The final board position in FEN notation
        result: The game result ('win', 'loss', 'draw')

    Returns:
        'checkmate' or 'stalemate' if detectable, None otherwise
    """
    if not fen:
        return None

    try:
        board = chess.Board(fen)

        # Check if the position has no legal moves
        if not any(board.legal_moves):
            if board.is_checkmate():
                return "checkmate"
            elif board.is_stalemate():
                return "stalemate"

        return None
    except Exception:
        return None
