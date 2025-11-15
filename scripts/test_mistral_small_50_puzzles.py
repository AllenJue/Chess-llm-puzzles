#!/usr/bin/env python3
"""
Test mistralai/mistral-small-24b-instruct-2501 on the first 50 puzzles.
This model was the only one that solved puzzle 0.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from main import load_environment, evaluate_puzzles, read_chess_puzzles_csv
from model_interface import ChessModelInterface
import pandas as pd

# The successful model from puzzle 0 testing
MODEL_NAME = "mistralai/mistral-small-24b-instruct-2501"


def test_model_on_50_puzzles():
    """Test the successful model on first 50 puzzles."""
    print("="*80)
    print(f"Testing {MODEL_NAME} on first 50 puzzles")
    print("="*80)
    
    # Load environment
    load_environment()
    
    # Get API config (using Anannas)
    api_key = os.getenv("ANANNAS_API_KEY")
    base_url = os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
    
    if not api_key:
        print("Error: ANANNAS_API_KEY not found in .env file")
        print("Please ensure ANANNAS_API_KEY is set in chess_puzzles/.env")
        return
    
    print(f"Using Anannas API")
    print(f"API key loaded: {bool(api_key)}")
    print(f"Base URL: {base_url}")
    print()
    
    # Create model interface
    model_interface = ChessModelInterface(
        api_key=api_key,
        model_name=MODEL_NAME,
        base_url=base_url,
        max_completion_tokens=640,
        default_temperature=0.1,
        retry_attempts=2,
    )
    
    # Load puzzles
    csv_file = os.path.join(parent_dir, "data", "input", "lichess_puzzles_with_pgn_1000.csv")
    df = read_chess_puzzles_csv(csv_file)
    
    print(f"Loaded {len(df)} puzzles from {csv_file}")
    print(f"Testing on puzzles 0-49 (first 50 puzzles)")
    print()
    
    # Evaluate puzzles
    df_results = evaluate_puzzles(
        df,
        model_interface=model_interface,
        max_puzzles=50,
        start_puzzle=0,
        planning_plies=0,
        api_delay=0.5,
    )
    
    # Calculate statistics
    total_puzzles = len(df_results)
    solved = df_results["puzzle_solved"].sum()
    accuracy = (solved / total_puzzles * 100) if total_puzzles > 0 else 0
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total puzzles tested: {total_puzzles}")
    print(f"Puzzles solved: {solved}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    
    # Show breakdown by correct moves
    print("Breakdown by correct moves:")
    move_counts = df_results["correct_moves"].value_counts().sort_index(ascending=False)
    for moves, count in move_counts.items():
        print(f"  {moves} correct moves: {count} puzzles")
    print()
    
    # Save results
    output_dir = parent_dir / "data" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = MODEL_NAME.replace("/", "_").replace(":", "_")
    output_file = output_dir / f"mistral_small_24b_50_puzzles_{timestamp}.csv"
    
    df_results.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")
    
    # Show some examples
    print("\n" + "="*80)
    print("Sample Results (first 10 puzzles):")
    print("="*80)
    sample_cols = ["PuzzleId", "puzzle_solved", "correct_moves", "total_moves", "single_model_move", "Moves"]
    print(df_results[sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    test_model_on_50_puzzles()

