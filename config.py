import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Bot configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

# Database
DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/chesstracker.db")
Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)

# Polling
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

# API tokens
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# API URLs
CHESSCOM_API_BASE = "https://api.chess.com/pub"
LICHESS_API_BASE = "https://lichess.org/api"

# Rate limiting (requests per minute)
CHESSCOM_RATE_LIMIT = 60  # Conservative limit
LICHESS_RATE_LIMIT = 15   # Without token

# Time control classifications (in seconds)
TIME_CONTROL_THRESHOLDS = {
    "bullet": (0, 180),      # < 3 minutes
    "blitz": (180, 600),     # 3-10 minutes
    "rapid": (600, 1800),    # 10-30 minutes
    "classical": (1800, float("inf")),  # > 30 minutes
}

# Platforms
class Platform:
    CHESSCOM = "chesscom"
    LICHESS = "lichess"

# Game results
class GameResult:
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"
