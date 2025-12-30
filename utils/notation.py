"""Chess notation parsing utilities for quiz answers."""

import re
from typing import Optional

import chess


def parse_user_move(user_input: str, board: chess.Board) -> Optional[chess.Move]:
    """
    Parse user's chess notation input into a Move object.

    Accepts various formats:
    - Standard algebraic notation: e4, Nf3, Qxd5, Bxe5
    - Castling: O-O, O-O-O, 0-0, 0-0-0
    - Pawn promotion: e8=Q, e8Q
    - With or without check/mate symbols: Qd8+, Qd8#, Qd8
    - With or without capture notation: Nxf3, Nf3
    - UCI notation: e2e4

    Args:
        user_input: The user's move input string
        board: The current board position

    Returns:
        chess.Move object if valid and legal, None otherwise
    """
    if not user_input or not user_input.strip():
        return None

    move_str = user_input.strip()

    # Handle common castling variations (0-0 -> O-O)
    move_str = re.sub(r"^0-0-0$", "O-O-O", move_str, flags=re.IGNORECASE)
    move_str = re.sub(r"^0-0$", "O-O", move_str, flags=re.IGNORECASE)

    # Normalize castling case
    if move_str.upper() in ("O-O", "O-O-O"):
        move_str = move_str.upper()

    # Remove annotations (+, #, !, ?)
    move_str = re.sub(r"[+#!?]+$", "", move_str)

    # Try parsing as standard algebraic notation
    try:
        return board.parse_san(move_str)
    except ValueError:
        pass

    # Try with uppercase piece letter (user might type "nf3" instead of "Nf3")
    if len(move_str) >= 2 and move_str[0].lower() in "nbrqk":
        move_str_upper = move_str[0].upper() + move_str[1:]
        try:
            return board.parse_san(move_str_upper)
        except ValueError:
            pass

    # Try as UCI notation (e2e4 format)
    try:
        move = chess.Move.from_uci(move_str.lower())
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    # Try UCI with promotion (e7e8q format)
    if len(move_str) == 5 and move_str[4].lower() in "qrbn":
        try:
            move = chess.Move.from_uci(move_str.lower())
            if move in board.legal_moves:
                return move
        except ValueError:
            pass

    return None


def normalize_san(san: str) -> str:
    """
    Normalize a SAN move string for comparison.

    Removes annotations and normalizes whitespace.

    Args:
        san: Standard algebraic notation string

    Returns:
        Normalized SAN string
    """
    # Remove annotations (+, #, !, ?)
    san = re.sub(r"[+#!?]+", "", san)
    return san.strip()


def format_move_for_display(move: chess.Move, board: chess.Board) -> str:
    """
    Format a move for display to users.

    Args:
        move: The chess move
        board: The board position BEFORE the move

    Returns:
        Human-readable SAN notation (e.g., "Nf3", "O-O", "e8=Q+")
    """
    return board.san(move)
