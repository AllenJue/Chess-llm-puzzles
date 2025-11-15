#!/usr/bin/env python3
"""
Script to rerun specific puzzles that had 'query1' KeyError in self-consistency mode.
This script will rerun only the puzzles that failed with this specific error.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from main import load_environment, evaluate_puzzles, read_chess_puzzles_csv, ChessSelfConsistency
from model_interface import ChessModelInterface

# Puzzles with 'query1' errors
QUERY1_ERROR_PUZZLES = {
    'gpt-3.5-turbo-instruct': ['k1hWM', '7LY8A', 'g7ODi', 'LePY2', 'msMhH'],
    'arcee-ai/afm-4.5b': [
        'f9PQx', 'RmvPy', 'uu1eQ', 'NZNGr', 'PsA2z', 'MgPSr', 'x0meF', 'k1hWM', 
        'fIvUy', 'ZlzMg', 'uMeJi', 'RkzY5', 'Anj44', 'zcgH1', 'hNgCd', 'B4iwI', 
        '7LY8A', 'hNedI', 'WmZxY', '2Z2JA', 'hZqbA', 'iIvSu', '073LS', 'RU2fn', 
        'g7ODi', 'UZABu', '2dCZ9', 's79rI', 'MyzF1', 'wDwLc', 'LePY2', 'tRRZ8', 
        'dsEdI', 'igtvn', 'msMhH', 'mCjnx', 'mO1IJ', '7b2mb'
    ]
}


def rerun_puzzles_for_model(model_name: str, puzzle_ids: list, csv_file: str = None):
    """Rerun specific puzzles for a model in self-consistency mode."""
    
    if csv_file is None:
        csv_file = "data/input/lichess_puzzles_with_pgn_1000.csv"
    
    print(f"\n{'='*80}")
    print(f"Rerunning {len(puzzle_ids)} puzzles for {model_name} in self-consistency mode")
    print(f"{'='*80}\n")
    
    # Load environment
    load_environment()
    
    # Read puzzles
    df_puzzles = read_chess_puzzles_csv(csv_file)
    
    # Filter to only the puzzles we want to rerun
    df_filtered = df_puzzles[df_puzzles['PuzzleId'].isin(puzzle_ids)].copy()
    
    if len(df_filtered) == 0:
        print(f"❌ No puzzles found with IDs: {puzzle_ids}")
        return
    
    print(f"Found {len(df_filtered)} puzzles to rerun:")
    print(f"  Puzzle IDs: {', '.join(df_filtered['PuzzleId'].tolist())}\n")
    
    # Create model interface
    model_interface = ChessModelInterface(model_name=model_name)
    
    # Create self-consistency instance
    debate = ChessSelfConsistency(
        model_name=model_name,
        temperature=0.1,
        openai_api_key=model_interface.api_key,
        base_url=model_interface.base_url
    )
    
    # Run evaluation
    output_file = f"data/test_results/self_consistency_50/rerun_query1_errors_{model_name.replace('/', '_')}.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Running evaluation...")
    df_results = evaluate_puzzles(
        df=df_filtered,
        debate=debate,
        max_puzzles=len(df_filtered),
        start_puzzle=0,
        planning_plies=0,
        api_delay=0.0
    )
    
    # Save results
    df_results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    print(f"   Puzzles rerun: {len(df_results)}")
    
    # Check for errors
    errors = df_results[df_results['error'].notna() & (df_results['error'] != '')]
    if len(errors) > 0:
        print(f"\n⚠️  {len(errors)} puzzles still have errors:")
        for idx, row in errors.iterrows():
            print(f"   {row['PuzzleId']}: {str(row['error'])[:100]}")
    else:
        print(f"\n✅ All puzzles completed without errors!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rerun puzzles with 'query1' KeyError")
    parser.add_argument("--model", type=str, required=True,
                       choices=['gpt-3.5-turbo-instruct', 'arcee-ai/afm-4.5b'],
                       help="Model to rerun")
    parser.add_argument("--csv-file", type=str,
                       default="data/input/lichess_puzzles_with_pgn_1000.csv",
                       help="Path to input CSV file")
    
    args = parser.parse_args()
    
    if args.model not in QUERY1_ERROR_PUZZLES:
        print(f"❌ Unknown model: {args.model}")
        print(f"Available models: {list(QUERY1_ERROR_PUZZLES.keys())}")
        sys.exit(1)
    
    puzzle_ids = QUERY1_ERROR_PUZZLES[args.model]
    rerun_puzzles_for_model(args.model, puzzle_ids, args.csv_file)


if __name__ == "__main__":
    main()

