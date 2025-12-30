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


# Performance settings (tune these for slower machines)
# Stockfish depth: lower = faster, less accurate (range: 1-20, default: 10)
# With NNUE, depth 10 provides excellent accuracy while being much faster than depth 12
STOCKFISH_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "10"))

# Stockfish hash table size in MB (default: 64MB, higher = better for repeated positions)
STOCKFISH_HASH_MB = int(os.getenv("STOCKFISH_HASH_MB", "64"))

# Stockfish threads per engine (default: 1, keep at 1 for parallel evaluation)
STOCKFISH_THREADS = int(os.getenv("STOCKFISH_THREADS", "1"))

# Number of Stockfish engines in pool (default: CPU count, 0 = auto)
STOCKFISH_POOL_SIZE = int(os.getenv("STOCKFISH_POOL_SIZE", "0")) or None

# Quiz-specific Stockfish depth (higher for better puzzle accuracy, default: 12)
# With NNUE, depth 12 provides excellent accuracy for blunder detection
QUIZ_STOCKFISH_DEPTH = int(os.getenv("QUIZ_STOCKFISH_DEPTH", "12"))

# Board size in pixels: smaller = faster rendering (default: 400)
BOARD_SIZE = int(os.getenv("BOARD_SIZE", "400"))

# Max parallel workers for evaluation (default: CPU count, set lower for weak CPUs)
MAX_EVAL_WORKERS = int(os.getenv("MAX_EVAL_WORKERS", "0")) or None  # 0 = auto

# Skip Stockfish and use material evaluation only (much faster, less accurate)
USE_MATERIAL_EVAL_ONLY = os.getenv("USE_MATERIAL_EVAL_ONLY", "false").lower() == "true"

# Disable video generation entirely (just show static board image)
DISABLE_VIDEO = os.getenv("DISABLE_VIDEO", "false").lower() == "true"

# Skip first N opening moves for evaluation (they're usually book moves)
SKIP_OPENING_MOVES = int(os.getenv("SKIP_OPENING_MOVES", "0"))

# Enable move/capture sounds in videos (requires ffmpeg)
ENABLE_VIDEO_AUDIO = os.getenv("ENABLE_VIDEO_AUDIO", "true").lower() == "true"

# Evaluation cache file path
EVAL_CACHE_PATH = os.getenv("EVAL_CACHE_PATH", "./data/eval_cache.json")
