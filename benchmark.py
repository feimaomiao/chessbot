#!/usr/bin/env python3
"""
Benchmark script for video generation and evaluation performance.

Tests:
- Video generation (uncached)
- Video generation (cached)
- Evaluation (uncached)
- Evaluation (cached)

Usage:
    python benchmark.py [--games N]
"""

import argparse
import asyncio
import statistics
import time
from typing import Callable

from api.lichess import LichessClient
from api.chesscom import ChessComClient
from config import DATABASE_PATH, Platform
from database import DatabaseManager
from utils.video import (
    StockfishEvaluator,
    generate_game_video,
    get_eval_cache,
    parse_pgn_positions,
)


async def fetch_pgns_from_db(max_games: int = 5) -> list[str]:
    """
    Fetch PGNs for games stored in the database.

    Args:
        max_games: Maximum number of games to fetch

    Returns:
        List of PGN strings
    """
    db = DatabaseManager(DATABASE_PATH)
    await db.connect()

    pgns = []
    lichess_client = LichessClient()
    chesscom_client = ChessComClient()

    try:
        # Get all tracked players
        players = await db.get_all_tracked_players()

        for player in players:
            if len(pgns) >= max_games:
                break

            # Get recent games for this player
            games = await db.get_recent_games(player.id, limit=max_games - len(pgns))

            for game in games:
                if len(pgns) >= max_games:
                    break

                try:
                    # Fetch PGN based on platform
                    if game.platform == Platform.LICHESS:
                        pgn = await lichess_client.get_game_pgn(game.game_id)
                    elif game.platform == Platform.CHESSCOM:
                        pgn = await chesscom_client.get_game_pgn(
                            player.username, game.game_id
                        )
                    else:
                        continue

                    if pgn:
                        # Validate PGN by parsing it
                        positions = parse_pgn_positions(pgn)
                        if len(positions) > 5:  # At least a few moves
                            pgns.append(pgn)
                            print(f"  Fetched game {game.game_id} ({len(positions)} positions)")

                except Exception as e:
                    print(f"  Failed to fetch {game.game_id}: {e}")
                    continue

    finally:
        await db.close()
        await lichess_client.close()
        await chesscom_client.close()

    return pgns


def benchmark(
    name: str,
    func: Callable[[], None],
    iterations: int = 3,
) -> dict:
    """
    Run a benchmark and return timing statistics.

    Args:
        name: Name of the benchmark
        func: Function to benchmark
        iterations: Number of iterations to run

    Returns:
        Dictionary with timing statistics
    """
    times = []

    for i in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}/{iterations}: {elapsed:.3f}s")

    return {
        "name": name,
        "iterations": iterations,
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_benchmarks(pgns: list[str]):
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("Chess Tracker Benchmark Suite")
    print("=" * 60)

    if not pgns:
        print("\nNo games found in database. Add some tracked players first.")
        return

    # Use first PGN for benchmarks
    pgn = pgns[0]
    positions = parse_pgn_positions(pgn)
    boards = [board for board, _ in positions]
    print(f"\nBenchmark game: {len(positions)} positions")

    # Check Stockfish availability
    evaluator = StockfishEvaluator()
    if not evaluator.available:
        print(
            "\nWARNING: Stockfish not available. "
            "Evaluation benchmarks will use material evaluation."
        )
    else:
        print(f"Stockfish available at depth {evaluator.depth}")

    cache = get_eval_cache()
    results = []

    # --- Evaluation Benchmarks ---
    print("\n" + "-" * 60)
    print("EVALUATION BENCHMARKS (parallel)")
    print("-" * 60)

    # Evaluation (uncached)
    print("\n[1] Evaluation (uncached)")

    def eval_uncached():
        cache.clear()
        evaluator.evaluate_positions(boards, parallel=True)

    results.append(benchmark("Evaluation (uncached)", eval_uncached))

    # Evaluation (cached) - cache is now warm from previous run
    print("\n[2] Evaluation (cached)")

    def eval_cached():
        evaluator.evaluate_positions(boards, parallel=True)

    # Warm the cache first
    cache.clear()
    evaluator.evaluate_positions(boards, parallel=True)

    results.append(benchmark("Evaluation (cached)", eval_cached))
    print(f"  Cache stats: {cache.get_stats()}")

    # --- Video Generation Benchmarks ---
    print("\n" + "-" * 60)
    print("VIDEO GENERATION BENCHMARKS")
    print("-" * 60)

    # Video generation (uncached)
    print("\n[3] Video generation (uncached)")

    def video_uncached():
        cache.clear()
        generate_game_video(pgn, use_stockfish=evaluator.available)

    results.append(
        benchmark("Video generation (uncached)", video_uncached, iterations=2)
    )

    # Video generation (cached) - cache is warm
    print("\n[4] Video generation (cached)")

    # Warm the cache
    cache.clear()
    generate_game_video(pgn, use_stockfish=evaluator.available)

    def video_cached():
        generate_game_video(pgn, use_stockfish=evaluator.available)

    results.append(benchmark("Video generation (cached)", video_cached, iterations=2))
    print(f"  Cache stats: {cache.get_stats()}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Benchmark':<35} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)

    for r in results:
        print(
            f"{r['name']:<35} {r['mean']:>9.3f}s {r['min']:>9.3f}s {r['max']:>9.3f}s"
        )

    # Calculate speedups
    print("\n" + "-" * 60)
    print("SPEEDUPS (cached vs uncached)")
    print("-" * 60)

    eval_speedup = (
        results[0]["mean"] / results[1]["mean"] if results[1]["mean"] > 0 else 0
    )
    video_speedup = (
        results[2]["mean"] / results[3]["mean"] if results[3]["mean"] > 0 else 0
    )

    print(f"Evaluation speedup:       {eval_speedup:>6.1f}x")
    print(f"Video generation speedup: {video_speedup:>6.1f}x")

    # Cleanup
    evaluator.close()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark video and evaluation performance")
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=3,
        help="Number of games to fetch from database (default: 3)"
    )
    args = parser.parse_args()

    print("Fetching games from database...")
    pgns = await fetch_pgns_from_db(max_games=args.games)
    print(f"Fetched {len(pgns)} valid game(s)\n")

    run_benchmarks(pgns)


if __name__ == "__main__":
    asyncio.run(main())
