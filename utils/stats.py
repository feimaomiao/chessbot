"""
Performance statistics tracking for video generation and evaluation.
"""

import json
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

STATS_FILE = Path("./data/performance_stats.json")


@dataclass
class OperationStats:
    """Statistics for a single operation type."""
    total_operations: int = 0
    total_positions: int = 0
    total_time_ms: float = 0
    min_time_per_position_ms: float = float("inf")
    max_time_per_position_ms: float = 0
    last_operation_time: Optional[str] = None

    @property
    def avg_time_per_position_ms(self) -> float:
        if self.total_positions == 0:
            return 0
        return self.total_time_ms / self.total_positions

    def record(self, positions: int, time_ms: float):
        """Record a completed operation."""
        self.total_operations += 1
        self.total_positions += positions
        self.total_time_ms += time_ms
        self.last_operation_time = datetime.utcnow().isoformat()

        if positions > 0:
            time_per_pos = time_ms / positions
            self.min_time_per_position_ms = min(self.min_time_per_position_ms, time_per_pos)
            self.max_time_per_position_ms = max(self.max_time_per_position_ms, time_per_pos)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_operations": self.total_operations,
            "total_positions": self.total_positions,
            "total_time_ms": self.total_time_ms,
            "min_time_per_position_ms": self.min_time_per_position_ms if self.min_time_per_position_ms != float("inf") else None,
            "max_time_per_position_ms": self.max_time_per_position_ms,
            "avg_time_per_position_ms": self.avg_time_per_position_ms,
            "last_operation_time": self.last_operation_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OperationStats":
        """Create from dictionary."""
        stats = cls()
        stats.total_operations = data.get("total_operations", 0)
        stats.total_positions = data.get("total_positions", 0)
        stats.total_time_ms = data.get("total_time_ms", 0)
        min_time = data.get("min_time_per_position_ms")
        stats.min_time_per_position_ms = min_time if min_time is not None else float("inf")
        stats.max_time_per_position_ms = data.get("max_time_per_position_ms", 0)
        stats.last_operation_time = data.get("last_operation_time")
        return stats


@dataclass
class PerformanceStats:
    """Container for all performance statistics."""
    evaluation: OperationStats = field(default_factory=OperationStats)
    video_generation: OperationStats = field(default_factory=OperationStats)
    quiz_evaluation: OperationStats = field(default_factory=OperationStats)
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    quiz_cache_hits: int = 0
    quiz_cache_misses: int = 0
    started_at: Optional[str] = None

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_cache_hits + self.total_cache_misses
        if total == 0:
            return 0.0
        return (self.total_cache_hits / total) * 100

    @property
    def quiz_cache_hit_rate(self) -> float:
        total = self.quiz_cache_hits + self.quiz_cache_misses
        if total == 0:
            return 0.0
        return (self.quiz_cache_hits / total) * 100

    def to_dict(self) -> dict:
        return {
            "evaluation": self.evaluation.to_dict(),
            "video_generation": self.video_generation.to_dict(),
            "quiz_evaluation": self.quiz_evaluation.to_dict(),
            "total_cache_hits": self.total_cache_hits,
            "total_cache_misses": self.total_cache_misses,
            "quiz_cache_hits": self.quiz_cache_hits,
            "quiz_cache_misses": self.quiz_cache_misses,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerformanceStats":
        stats = cls()
        stats.evaluation = OperationStats.from_dict(data.get("evaluation", {}))
        stats.video_generation = OperationStats.from_dict(data.get("video_generation", {}))
        stats.quiz_evaluation = OperationStats.from_dict(data.get("quiz_evaluation", {}))
        stats.total_cache_hits = data.get("total_cache_hits", 0)
        stats.total_cache_misses = data.get("total_cache_misses", 0)
        stats.quiz_cache_hits = data.get("quiz_cache_hits", 0)
        stats.quiz_cache_misses = data.get("quiz_cache_misses", 0)
        stats.started_at = data.get("started_at")
        return stats


class StatsTracker:
    """Thread-safe performance statistics tracker."""

    def __init__(self, stats_file: Path = STATS_FILE):
        self._stats_file = stats_file
        self._lock = threading.Lock()
        self._stats = self._load_stats()

    def _load_stats(self) -> PerformanceStats:
        """Load stats from file or create new."""
        if self._stats_file.exists():
            try:
                with open(self._stats_file, "r") as f:
                    data = json.load(f)
                    return PerformanceStats.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load stats file: {e}")

        stats = PerformanceStats()
        stats.started_at = datetime.utcnow().isoformat()
        return stats

    def _save_stats(self):
        """Save stats to file."""
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._stats_file, "w") as f:
                json.dump(self._stats.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats file: {e}")

    def record_evaluation(self, positions: int, time_ms: float, cache_hits: int):
        """Record an evaluation operation."""
        with self._lock:
            self._stats.evaluation.record(positions, time_ms)
            self._stats.total_cache_hits += cache_hits
            self._stats.total_cache_misses += positions - cache_hits
            self._save_stats()

    def record_video_generation(self, positions: int, time_ms: float, cache_hits: int):
        """Record a video generation operation with cache stats."""
        with self._lock:
            self._stats.video_generation.record(positions, time_ms)
            self._stats.total_cache_hits += cache_hits
            self._stats.total_cache_misses += positions - cache_hits
            self._save_stats()

    def record_video_timing(self, positions: int, time_ms: float):
        """Record video generation timing only (no cache stats)."""
        with self._lock:
            self._stats.video_generation.record(positions, time_ms)
            self._save_stats()

    def record_quiz_evaluation(self, positions: int, time_ms: float, cache_hits: int = 0):
        """Record a quiz evaluation operation."""
        with self._lock:
            self._stats.quiz_evaluation.record(positions, time_ms)
            self._stats.quiz_cache_hits += cache_hits
            self._stats.quiz_cache_misses += positions - cache_hits
            self._save_stats()

    def get_stats(self) -> PerformanceStats:
        """Get current stats."""
        with self._lock:
            return self._stats

    def reset(self):
        """Reset all stats."""
        with self._lock:
            self._stats = PerformanceStats()
            self._stats.started_at = datetime.utcnow().isoformat()
            self._save_stats()


# Global tracker instance
_stats_tracker: Optional[StatsTracker] = None


def get_stats_tracker() -> StatsTracker:
    """Get or create the global stats tracker."""
    global _stats_tracker
    if _stats_tracker is None:
        _stats_tracker = StatsTracker()
    return _stats_tracker


def get_system_specs() -> dict:
    """Get system specifications."""
    try:
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()

        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor() or "Unknown",
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": int(cpu_freq.current) if cpu_freq else None,
            "memory_total_gb": round(memory.total / (1024**3), 1),
            "memory_available_gb": round(memory.available / (1024**3), 1),
            "python_version": platform.python_version(),
        }
    except Exception as e:
        logger.error(f"Error getting system specs: {e}")
        return {
            "platform": platform.system(),
            "error": str(e),
        }


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
