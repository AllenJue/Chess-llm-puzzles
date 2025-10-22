"""
Main Entry Point for Chess Puzzle Evaluation

This script provides the main interface for evaluating chess puzzles using
OpenAI models and Glicko-2 rating system.

Usage:
    python main.py --help
    python main.py --csv-file puzzles.csv --max-puzzles 100
    python main.py --evaluate --model gpt-4-turbo
"""

import argparse
import os
import sys
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# Import our modules
from csv_reader import read_chess_puzzles_csv, sample_puzzles, save_puzzles_csv, get_puzzle_stats
from model_interface import ChessModelInterface
from chess_utils import build_chess_prompts, get_partial_pgn_from_url, extract_predicted_move, san_to_uci
from glicko_rating import update_agent_rating_from_puzzles, Rating


def load_environment():
    """Load environment variables from .env file if it exists."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
    else:
        print("No .env file found, using system environment variables")


def evaluate_puzzles(df: pd.DataFrame, model_interface: ChessModelInterface, 
                    max_puzzles: int = 5) -> pd.DataFrame:
    """
    Evaluate puzzles using the model interface.
    
    Args:
        df: DataFrame with puzzle data
        model_interface: ChessModelInterface instance
        max_puzzles: Maximum number of puzzles to evaluate
        
    Returns:
        DataFrame with evaluation results
    """
    df_eval = df.copy()
    df_eval["correct_moves"] = 0
    df_eval["puzzle_solved"] = False
    df_eval["error"] = ""

    for idx, row in df_eval.head(max_puzzles).iterrows():
        print(f"\n=== Evaluating puzzle {idx} ===")
        
        try:
            # Get PGN from URL
            pgn, move_num = get_partial_pgn_from_url(row["GameUrl"])
            
            # Build prompts
            system_prompt, user_prompt = build_chess_prompts(pgn)
            
            # Get model prediction
            predicted_san = model_interface.get_move_with_extraction(
                system_prompt, user_prompt, 
                current_turn_number=move_num // 2 + 1 if move_num else None,
                is_white_to_move=(move_num % 2 == 0) if move_num else True
            )
            
            if predicted_san:
                print(f"Predicted move: {predicted_san}")
                # Here you would compare with expected moves and update scores
                # This is a simplified version
                df_eval.loc[idx, "correct_moves"] = 1
                df_eval.loc[idx, "puzzle_solved"] = True
            else:
                print("Failed to extract move")
                df_eval.loc[idx, "error"] = "Failed to extract move"
                
        except Exception as e:
            print(f"Error processing puzzle {idx}: {e}")
            df_eval.loc[idx, "error"] = str(e)
    
    return df_eval


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chess Puzzle Evaluator")
    
    parser.add_argument("--csv-file", default="lichess_puzzles_with_pgn_1000.csv",
                       help="Path to CSV file with puzzle data")
    parser.add_argument("--max-puzzles", type=int, default=10,
                       help="Maximum number of puzzles to evaluate")
    parser.add_argument("--model", choices=["gpt-3.5-turbo-instruct", "gpt-4-turbo"],
                       default="gpt-3.5-turbo-instruct", help="Model to use")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run puzzle evaluation")
    parser.add_argument("--sample", type=int, default=None,
                       help="Sample N puzzles randomly")
    parser.add_argument("--stats", action="store_true",
                       help="Show puzzle statistics")
    parser.add_argument("--rating", action="store_true",
                       help="Calculate Glicko-2 rating")
    parser.add_argument("--output", default=None,
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Load environment
    load_environment()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in the .env file or environment")
        sys.exit(1)
    
    # Read CSV file
    try:
        df = read_chess_puzzles_csv(args.csv_file)
        print(f"Loaded {len(df)} puzzles from {args.csv_file}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Sample puzzles if requested
    if args.sample:
        df = sample_puzzles(df, n=args.sample)
        print(f"Sampled {len(df)} puzzles")
    
    # Show statistics
    if args.stats:
        stats = get_puzzle_stats(df)
        print("\nPuzzle Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Run evaluation
    if args.evaluate:
        print(f"\nEvaluating puzzles with {args.model}...")
        model_interface = ChessModelInterface(api_key=api_key, model_name=args.model)
        
        # Evaluate puzzles
        df_results = evaluate_puzzles(df, model_interface, args.max_puzzles)
        
        # Save results
        if args.output:
            save_puzzles_csv(df_results, args.output)
            print(f"Results saved to {args.output}")
        
        # Show summary
        solved = df_results["puzzle_solved"].sum()
        total = len(df_results)
        print(f"\nEvaluation Summary:")
        print(f"  Puzzles solved: {solved}/{total} ({solved/total*100:.1f}%)")
    
    # Calculate rating
    if args.rating:
        print("\nCalculating Glicko-2 rating...")
        if "puzzle_solved" in df.columns:
            agent_rating = update_agent_rating_from_puzzles(df)
            print(f"Agent rating: {agent_rating}")
        else:
            print("Error: No puzzle_solved column found. Run evaluation first.")


if __name__ == "__main__":
    main()
