import io
import logging
from datetime import datetime
from typing import Optional

import chess
import chess.pgn

from config import CHESSCOM_API_BASE, CHESSCOM_RATE_LIMIT, GameResult
from utils.helpers import parse_time_control
from .base import BaseChessClient, GameData

logger = logging.getLogger(__name__)


class ChessComClient(BaseChessClient):
    """Chess.com API client."""

    PLATFORM = "chesscom"

    def __init__(self):
        super().__init__(rate_limit=CHESSCOM_RATE_LIMIT)
        self.base_url = CHESSCOM_API_BASE

    async def validate_player(self, username: str) -> bool:
        """Check if a player exists on Chess.com."""
        url = f"{self.base_url}/player/{username.lower()}"
        data = await self._get(url)
        return data is not None

    async def get_recent_games(
        self, username: str, since: Optional[datetime] = None
    ) -> list[GameData]:
        """Get recent games for a player from Chess.com."""
        username = username.lower()
        games = []

        # Get current month's games archive
        now = datetime.utcnow()
        archive_url = f"{self.base_url}/player/{username}/games/{now.year}/{now.month:02d}"

        data = await self._get(archive_url)
        if not data or "games" not in data:
            return games

        for game_data in data["games"]:
            try:
                game = self._parse_game(username, game_data)
                if game:
                    # Filter by time if specified (use <= to exclude the last known game)
                    if since and game.played_at <= since:
                        continue
                    games.append(game)
            except Exception as e:
                logger.error(f"Error parsing game: {e}")
                continue

        return games

    def _parse_game(self, username: str, data: dict) -> Optional[GameData]:
        """Parse a Chess.com game into GameData."""
        try:
            # Determine if player is white or black
            white = data.get("white", {})
            black = data.get("black", {})

            is_white = white.get("username", "").lower() == username
            player_data = white if is_white else black
            opponent_data = black if is_white else white

            # Parse result
            player_result = player_data.get("result", "")
            result = self._parse_result(player_result)

            # Parse time control
            time_control_str = data.get("time_control", "")
            time_control, tc_display = parse_time_control(time_control_str)

            # Get ratings (rating_change will be calculated from history later)
            rating_after = player_data.get("rating", 0)
            opponent_rating = opponent_data.get("rating", 0)
            rating_change = 0  # Will be calculated from game history

            # Parse timestamp
            end_time = data.get("end_time", 0)
            played_at = datetime.fromtimestamp(end_time)

            # Parse PGN to get final FEN
            final_fen = self._get_final_fen(data.get("pgn", ""))

            # Parse opening from eco URL
            opening_name, opening_eco = self._parse_opening(data)

            return GameData(
                game_id=data.get("uuid", str(end_time)),
                platform=self.PLATFORM,
                time_control=time_control,
                time_control_display=tc_display,
                result=result,
                player_color="white" if is_white else "black",
                rating_after=rating_after,
                rating_change=rating_change,
                opponent=opponent_data.get("username", "Unknown"),
                opponent_rating=opponent_rating,
                played_at=played_at,
                game_url=data.get("url", ""),
                final_fen=final_fen,
                opening_name=opening_name,
                opening_eco=opening_eco,
            )
        except Exception as e:
            logger.error(f"Error parsing Chess.com game: {e}")
            return None

    def _parse_opening(self, data: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Parse opening info from Chess.com game data.

        Chess.com provides an 'eco' field with a URL like:
        https://www.chess.com/openings/Sicilian-Defense-Open-2...Nf6-3.d4

        Returns:
            Tuple of (opening_name, eco_code)
        """
        eco_url = data.get("eco")
        if not eco_url:
            return None, None

        try:
            # Extract opening name from URL path
            # URL format: https://www.chess.com/openings/Opening-Name-Variation...
            path = eco_url.split("/openings/")[-1] if "/openings/" in eco_url else ""
            if not path:
                return None, None

            # Convert URL path to readable name
            # "Sicilian-Defense-Open-2...Nf6" -> "Sicilian Defense"
            # Take only the main opening name (before move numbers or long variations)
            parts = path.split("-")

            # Find where the variation/moves start (contains numbers or is too long)
            name_parts = []
            for part in parts:
                # Stop at move numbers like "2...Nf6" or "3.d4"
                if any(c.isdigit() for c in part) and ("." in part or len(part) > 10):
                    break
                name_parts.append(part)

            # Limit to reasonable opening name length
            if len(name_parts) > 5:
                name_parts = name_parts[:5]

            opening_name = " ".join(name_parts) if name_parts else None

            # ECO code is not directly available from the URL
            # We could parse from PGN headers if needed, but for now return None
            opening_eco = None

            return opening_name, opening_eco
        except Exception as e:
            logger.error(f"Error parsing opening from {eco_url}: {e}")
            return None, None

    def _parse_result(self, result_str: str) -> str:
        """Convert Chess.com result string to standardized result."""
        win_results = {"win"}
        loss_results = {
            "checkmated", "timeout", "resigned", "lose",
            "abandoned", "kingofthehill", "threecheck",
        }
        draw_results = {
            "agreed", "repetition", "stalemate", "insufficient",
            "50move", "timevsinsufficient", "draw",
        }

        result_lower = result_str.lower()
        if result_lower in win_results:
            return GameResult.WIN
        elif result_lower in loss_results:
            return GameResult.LOSS
        elif result_lower in draw_results:
            return GameResult.DRAW
        else:
            # Default based on common patterns
            if "win" in result_lower:
                return GameResult.WIN
            return GameResult.LOSS

    def _get_final_fen(self, pgn_str: str) -> Optional[str]:
        """Parse PGN and return the final board position as FEN."""
        if not pgn_str:
            return None

        try:
            pgn = chess.pgn.read_game(io.StringIO(pgn_str))
            if pgn is None:
                return None

            board = pgn.board()
            for move in pgn.mainline_moves():
                board.push(move)

            return board.fen()
        except Exception as e:
            logger.error(f"Error parsing PGN: {e}")
            return None

    async def get_player_stats(self, username: str) -> Optional[dict]:
        """Get player statistics from Chess.com."""
        url = f"{self.base_url}/player/{username.lower()}/stats"
        return await self._get(url)

    async def get_game_pgn(self, username: str, game_id: str) -> Optional[str]:
        """
        Fetch the PGN for a specific game.

        Chess.com doesn't have a direct game-by-ID endpoint, so we need to
        search through archives. The game_id is the UUID from the game.

        Args:
            username: The player's username (needed to find the game)
            game_id: The Chess.com game UUID

        Returns:
            PGN string or None if not found
        """
        username = username.lower()

        # Get archives list
        archives_url = f"{self.base_url}/player/{username}/games/archives"
        archives_data = await self._get(archives_url)

        if not archives_data or "archives" not in archives_data:
            return None

        # Search recent archives (most recent first)
        for archive_url in reversed(archives_data["archives"][-3:]):  # Last 3 months
            data = await self._get(archive_url)
            if not data or "games" not in data:
                continue

            for game in data["games"]:
                if game.get("uuid") == game_id:
                    return game.get("pgn")

        return None

    async def get_games_for_analysis(
        self, username: str, max_games: int = 1000
    ) -> list[GameData]:
        """
        Fetch a large number of games for analysis.

        Fetches multiple monthly archives to collect up to max_games.

        Args:
            username: The player's username
            max_games: Maximum number of games to fetch (default 1000)

        Returns:
            List of GameData, sorted by played_at (newest first)
        """
        username = username.lower()
        games = []

        # Get archives list
        archives_url = f"{self.base_url}/player/{username}/games/archives"
        archives_data = await self._get(archives_url)

        if not archives_data or "archives" not in archives_data:
            return games

        # Process archives from most recent to oldest
        for archive_url in reversed(archives_data["archives"]):
            if len(games) >= max_games:
                break

            data = await self._get(archive_url)
            if not data or "games" not in data:
                continue

            for game_data in reversed(data["games"]):  # Newest first within month
                if len(games) >= max_games:
                    break

                try:
                    game = self._parse_game(username, game_data)
                    if game:
                        games.append(game)
                except Exception as e:
                    logger.error(f"Error parsing game for analysis: {e}")
                    continue

        return games
