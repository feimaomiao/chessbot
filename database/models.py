from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Guild:
    id: int  # Discord guild ID
    notification_channel_id: Optional[int] = None
    summary_channel_id: Optional[int] = None
    summary_time: str = "00:00"  # HH:MM UTC (midnight)


@dataclass
class TrackedPlayer:
    id: Optional[int]  # Auto-increment
    guild_id: int
    platform: str  # 'chesscom' or 'lichess'
    username: str
    display_name: Optional[str] = None
    discord_user_id: Optional[int] = None  # Linked Discord member to ping
    added_by: Optional[int] = None  # Discord user ID who added
    added_at: Optional[datetime] = None

    def __post_init__(self):
        if self.added_at is None:
            self.added_at = datetime.utcnow()


@dataclass
class Game:
    id: Optional[int]  # Auto-increment
    player_id: int  # FK to TrackedPlayer
    game_id: str  # Platform's game ID
    platform: str
    time_control: str  # bullet/blitz/rapid/classical
    time_control_display: str  # e.g., "3+0"
    result: str  # win/loss/draw
    player_color: str  # "white" or "black"
    rating_after: int
    rating_change: int
    opponent: str
    opponent_rating: int
    played_at: datetime
    game_url: str
    final_fen: Optional[str] = None  # Final board position
    notified: bool = False
    accuracy: Optional[float] = None  # Player's accuracy percentage (0-100)
