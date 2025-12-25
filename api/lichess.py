import json
import logging
from datetime import datetime
from typing import Optional

from config import LICHESS_API_BASE, LICHESS_RATE_LIMIT, LICHESS_TOKEN, GameResult
from utils.helpers import parse_time_control
from .base import BaseChessClient, GameData

logger = logging.getLogger(__name__)


class LichessClient(BaseChessClient):
    """Lichess API client."""

    PLATFORM = "lichess"

    def __init__(self):
        super().__init__(rate_limit=LICHESS_RATE_LIMIT)
        self.base_url = LICHESS_API_BASE
        self.token = LICHESS_TOKEN

    def _get_headers(self) -> dict:
        """Get headers including auth token if available."""
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def validate_player(self, username: str) -> bool:
        """Check if a player exists on Lichess."""
        url = f"{self.base_url}/user/{username}"
        data = await self._get(url, headers=self._get_headers())
        return data is not None

    async def get_recent_games(
        self, username: str, since: Optional[datetime] = None
    ) -> list[GameData]:
        """Get recent games for a player from Lichess."""
        games = []

        # Build URL with parameters
        url = f"{self.base_url}/games/user/{username}"
        params = ["max=30", "pgnInJson=false", "lastFen=true", "opening=true"]

        if since:
            # Lichess uses milliseconds
            since_ms = int(since.timestamp() * 1000)
            params.append(f"since={since_ms}")

        url = f"{url}?{'&'.join(params)}"

        await self._ensure_session()
        await self._rate_limit_wait()

        try:
            headers = self._get_headers()
            headers["Accept"] = "application/x-ndjson"

            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Lichess API error: {response.status}")
                    return games

                # Lichess returns newline-delimited JSON
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        game_data = json.loads(line)
                        game = self._parse_game(username, game_data)
                        if game:
                            games.append(game)
                    except Exception as e:
                        logger.error(f"Error parsing game line: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error fetching Lichess games: {e}")

        return games

    def _parse_game(self, username: str, data: dict) -> Optional[GameData]:
        """Parse a Lichess game into GameData."""
        try:
            players = data.get("players", {})
            white = players.get("white", {})
            black = players.get("black", {})

            # Determine if player is white or black
            white_user = white.get("user", {}).get("name", "").lower()
            is_white = white_user == username.lower()

            player_data = white if is_white else black
            opponent_data = black if is_white else white

            # Get opponent info
            opponent_user = opponent_data.get("user", {})

            # Parse result
            winner = data.get("winner")
            if winner is None:
                result = GameResult.DRAW
            elif (winner == "white" and is_white) or (winner == "black" and not is_white):
                result = GameResult.WIN
            else:
                result = GameResult.LOSS

            # Parse time control - Lichess uses speed categories
            speed = data.get("speed", "unknown")
            time_control, tc_display = parse_time_control(speed)

            # Get clock info for display
            clock = data.get("clock", {})
            if clock:
                initial = clock.get("initial", 0)
                increment = clock.get("increment", 0)
                tc_display = f"{initial // 60}+{increment}"

            # Rating info
            rating_after = player_data.get("rating", 0)
            rating_diff = player_data.get("ratingDiff", 0)

            # Parse timestamp
            created_at = data.get("createdAt", 0)
            if created_at > 1e12:  # Milliseconds
                created_at = created_at / 1000
            played_at = datetime.fromtimestamp(created_at)

            game_id = data.get("id", "")

            # Get final FEN (requires lastFen=true in request)
            final_fen = data.get("lastFen")

            # Parse opening info (requires opening=true in request)
            opening_data = data.get("opening", {})
            opening_name = opening_data.get("name") if opening_data else None
            opening_eco = opening_data.get("eco") if opening_data else None

            return GameData(
                game_id=game_id,
                platform=self.PLATFORM,
                time_control=time_control,
                time_control_display=tc_display,
                result=result,
                player_color="white" if is_white else "black",
                rating_after=rating_after,
                rating_change=rating_diff,
                opponent=opponent_user.get("name", "Anonymous"),
                opponent_rating=opponent_data.get("rating", 0),
                played_at=played_at,
                game_url=f"https://lichess.org/{game_id}",
                final_fen=final_fen,
                opening_name=opening_name,
                opening_eco=opening_eco,
            )
        except Exception as e:
            logger.error(f"Error parsing Lichess game: {e}")
            return None

    async def get_player_info(self, username: str) -> Optional[dict]:
        """Get player profile information from Lichess."""
        url = f"{self.base_url}/user/{username}"
        return await self._get(url, headers=self._get_headers())

    async def get_game_pgn(self, game_id: str) -> Optional[str]:
        """
        Fetch the PGN for a specific game.

        Args:
            game_id: The Lichess game ID (e.g., "abcd1234")

        Returns:
            PGN string or None if fetch fails
        """
        url = f"{self.base_url}/game/export/{game_id}"

        await self._ensure_session()
        await self._rate_limit_wait()

        try:
            headers = self._get_headers()
            headers["Accept"] = "application/x-chess-pgn"

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.error(f"Lichess game fetch error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching Lichess game PGN: {e}")
            return None

    async def get_games_for_analysis(
        self, username: str, max_games: int = 1000
    ) -> list[GameData]:
        """
        Fetch a large number of games for analysis.

        Args:
            username: The player's username
            max_games: Maximum number of games to fetch (default 1000)

        Returns:
            List of GameData, sorted by played_at (newest first)
        """
        games = []

        # Build URL with parameters for bulk fetch
        url = f"{self.base_url}/games/user/{username}"
        params = [
            f"max={max_games}",
            "pgnInJson=false",
            "lastFen=true",
            "opening=true",
        ]

        url = f"{url}?{'&'.join(params)}"

        await self._ensure_session()
        await self._rate_limit_wait()

        try:
            headers = self._get_headers()
            headers["Accept"] = "application/x-ndjson"

            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Lichess API error: {response.status}")
                    return games

                # Lichess returns newline-delimited JSON
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        game_data = json.loads(line)
                        game = self._parse_game(username, game_data)
                        if game:
                            games.append(game)
                    except Exception as e:
                        logger.error(f"Error parsing game for analysis: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error fetching Lichess games for analysis: {e}")

        return games
