import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GameData:
    """Standardized game data from any platform."""
    game_id: str
    platform: str
    time_control: str
    time_control_display: str
    result: str  # win/loss/draw
    player_color: str  # "white" or "black"
    rating_after: int
    rating_change: int
    opponent: str
    opponent_rating: int
    played_at: datetime
    game_url: str
    final_fen: Optional[str] = None  # Final board position
    opening_name: Optional[str] = None  # Opening name (e.g., "Sicilian Defense")
    opening_eco: Optional[str] = None  # ECO code (e.g., "B20")
    termination: Optional[str] = None  # How game ended: checkmate, timeout, resign, aborted, agreed, stalemate, repetition


class BaseChessClient(ABC):
    """Base class for chess platform API clients."""

    def __init__(self, rate_limit: int = 30):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit = rate_limit
        self._request_times: list[float] = []

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _rate_limit_wait(self):
        """Wait if necessary to respect rate limits."""
        now = asyncio.get_event_loop().time()
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        self._request_times.append(now)

    async def _get(self, url: str, headers: Optional[dict] = None) -> Optional[dict]:
        """Make a GET request with rate limiting."""
        await self._ensure_session()
        await self._rate_limit_wait()

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    return None
                else:
                    logger.error(f"API error {response.status}: {url}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            return None

    @abstractmethod
    async def validate_player(self, username: str) -> bool:
        """Check if a player exists on the platform."""
        pass

    @abstractmethod
    async def get_recent_games(
        self, username: str, since: Optional[datetime] = None
    ) -> list[GameData]:
        """Get recent games for a player."""
        pass
