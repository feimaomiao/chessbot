"""
Analysis service for computing statistics from game data.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from api.base import GameData
from config import GameResult

logger = logging.getLogger(__name__)


@dataclass
class OpeningStats:
    """Statistics for a single opening."""
    name: str
    eco: Optional[str]
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate win rate as a percentage."""
        if self.games == 0:
            return 0.0
        return (self.wins / self.games) * 100


@dataclass
class AnalysisResult:
    """Complete analysis result for a player's games."""
    total_games: int
    date_range_start: Optional[datetime]
    date_range_end: Optional[datetime]

    # Opening stats (top 5 each)
    top_openings_white: list[OpeningStats] = field(default_factory=list)
    top_openings_black: list[OpeningStats] = field(default_factory=list)

    # Color stats
    white_games: int = 0
    white_wins: int = 0
    white_losses: int = 0
    white_draws: int = 0
    black_games: int = 0
    black_wins: int = 0
    black_losses: int = 0
    black_draws: int = 0

    # Rating progression
    starting_rating: int = 0
    ending_rating: int = 0
    rating_high: int = 0
    rating_low: int = 0

    # Accuracy stats
    avg_accuracy: Optional[float] = None
    games_with_accuracy: int = 0

    # Termination stats
    termination_checkmate: int = 0
    termination_timeout: int = 0
    termination_resign: int = 0
    termination_stalemate: int = 0
    termination_repetition: int = 0
    termination_agreed: int = 0
    termination_aborted: int = 0
    termination_unknown: int = 0

    @property
    def rating_change(self) -> int:
        """Total rating change over the period."""
        return self.ending_rating - self.starting_rating

    @property
    def white_win_rate(self) -> float:
        """Win rate as white."""
        if self.white_games == 0:
            return 0.0
        return (self.white_wins / self.white_games) * 100

    @property
    def black_win_rate(self) -> float:
        """Win rate as black."""
        if self.black_games == 0:
            return 0.0
        return (self.black_wins / self.black_games) * 100


def analyze_games(games: list[GameData]) -> AnalysisResult:
    """
    Analyze a list of games and compute statistics.

    Args:
        games: List of GameData objects (expected to be sorted newest first)

    Returns:
        AnalysisResult with computed statistics
    """
    if not games:
        return AnalysisResult(
            total_games=0,
            date_range_start=None,
            date_range_end=None,
        )

    # Sort games by date (oldest first for rating progression)
    sorted_games = sorted(games, key=lambda g: g.played_at)

    # Date range
    date_range_start = sorted_games[0].played_at
    date_range_end = sorted_games[-1].played_at

    # Initialize counters
    white_games = white_wins = white_losses = white_draws = 0
    black_games = black_wins = black_losses = black_draws = 0

    # Termination counters
    termination_counts = {
        "checkmate": 0,
        "timeout": 0,
        "resign": 0,
        "stalemate": 0,
        "repetition": 0,
        "agreed": 0,
        "aborted": 0,
        "unknown": 0,
    }

    # Opening tracking: {opening_name: OpeningStats}
    openings_white: dict[str, OpeningStats] = defaultdict(
        lambda: OpeningStats(name="", eco=None)
    )
    openings_black: dict[str, OpeningStats] = defaultdict(
        lambda: OpeningStats(name="", eco=None)
    )

    # Rating tracking
    ratings = []

    for game in sorted_games:
        # Track ratings
        if game.rating_after > 0:
            ratings.append(game.rating_after)

        # Track termination
        termination = getattr(game, 'termination', None) or "unknown"
        if termination in termination_counts:
            termination_counts[termination] += 1
        else:
            termination_counts["unknown"] += 1

        # Color stats
        is_white = game.player_color == "white"
        is_win = game.result == GameResult.WIN
        is_loss = game.result == GameResult.LOSS
        is_draw = game.result == GameResult.DRAW

        if is_white:
            white_games += 1
            if is_win:
                white_wins += 1
            elif is_loss:
                white_losses += 1
            else:
                white_draws += 1
        else:
            black_games += 1
            if is_win:
                black_wins += 1
            elif is_loss:
                black_losses += 1
            else:
                black_draws += 1

        # Opening stats
        if game.opening_name:
            # Normalize opening name to just the main opening
            opening_name = _normalize_opening_name(game.opening_name)

            if is_white:
                if opening_name not in openings_white:
                    openings_white[opening_name] = OpeningStats(
                        name=opening_name,
                        eco=game.opening_eco,
                    )
                stats = openings_white[opening_name]
            else:
                if opening_name not in openings_black:
                    openings_black[opening_name] = OpeningStats(
                        name=opening_name,
                        eco=game.opening_eco,
                    )
                stats = openings_black[opening_name]

            stats.games += 1
            if is_win:
                stats.wins += 1
            elif is_loss:
                stats.losses += 1
            else:
                stats.draws += 1

    # Get top 5 openings by game count
    top_white = sorted(
        openings_white.values(),
        key=lambda x: x.games,
        reverse=True
    )[:5]

    top_black = sorted(
        openings_black.values(),
        key=lambda x: x.games,
        reverse=True
    )[:5]

    # Rating stats
    starting_rating = ratings[0] if ratings else 0
    ending_rating = ratings[-1] if ratings else 0
    rating_high = max(ratings) if ratings else 0
    rating_low = min(ratings) if ratings else 0

    return AnalysisResult(
        total_games=len(games),
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        top_openings_white=top_white,
        top_openings_black=top_black,
        white_games=white_games,
        white_wins=white_wins,
        white_losses=white_losses,
        white_draws=white_draws,
        black_games=black_games,
        black_wins=black_wins,
        black_losses=black_losses,
        black_draws=black_draws,
        starting_rating=starting_rating,
        ending_rating=ending_rating,
        rating_high=rating_high,
        rating_low=rating_low,
        termination_checkmate=termination_counts["checkmate"],
        termination_timeout=termination_counts["timeout"],
        termination_resign=termination_counts["resign"],
        termination_stalemate=termination_counts["stalemate"],
        termination_repetition=termination_counts["repetition"],
        termination_agreed=termination_counts["agreed"],
        termination_aborted=termination_counts["aborted"],
        termination_unknown=termination_counts["unknown"],
    )


def _normalize_opening_name(name: str) -> str:
    """
    Normalize opening name to group variations together.

    For example:
    - "Sicilian Defense: Najdorf Variation" -> "Sicilian Defense"
    - "Italian Game: Two Knights Defense" -> "Italian Game"
    """
    # Split on common delimiters
    for delimiter in [":", ",", "with"]:
        if delimiter in name:
            name = name.split(delimiter)[0].strip()

    return name


def calculate_accuracy_stats(games) -> tuple[Optional[float], int]:
    """
    Calculate average accuracy from a list of database Game objects.

    Args:
        games: List of Game objects with accuracy field

    Returns:
        Tuple of (average_accuracy, games_with_accuracy_count)
    """
    accuracies = [g.accuracy for g in games if g.accuracy is not None]

    if not accuracies:
        return None, 0

    avg = sum(accuracies) / len(accuracies)
    return round(avg, 1), len(accuracies)
