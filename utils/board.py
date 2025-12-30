import io
import logging
from typing import Optional

import chess
import chess.svg
import cairosvg
from discord import File as DiscordFile

logger = logging.getLogger(__name__)


def render_board_png(fen: str, size: int = 400, flipped: bool = False) -> Optional[bytes]:
    """
    Render a chess board from FEN to PNG bytes.

    Args:
        fen: FEN string representing the board position
        size: Size of the output image in pixels
        flipped: If True, render from Black's perspective (a8 at bottom-left)

    Returns:
        PNG image as bytes, or None if rendering fails
    """
    if not fen:
        return None

    try:
        board = chess.Board(fen)

        # Generate SVG
        svg_data = chess.svg.board(
            board,
            size=size,
            flipped=flipped,
            coordinates=True,
            colors={
                "square light": "#f0d9b5",
                "square dark": "#b58863",
                "margin": "#212121",
                "coord": "#e0e0e0",
            },
        )

        # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))

        return png_data

    except Exception as e:
        logger.error(f"Error rendering board: {e}")
        return None


def get_board_discord_file(fen: str, filename: str = "board.png", flipped: bool = False):
    """
    Create a Discord file object from a FEN position.

    Args:
        fen: FEN string representing the board position
        filename: Name for the file attachment
        flipped: If True, render from Black's perspective

    Returns:
        discord.File object or None if rendering fails
    """
    png_data = render_board_png(fen, flipped=flipped)
    if png_data is None:
        return None

    return DiscordFile(io.BytesIO(png_data), filename=filename)
