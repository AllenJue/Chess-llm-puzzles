#!/usr/bin/env python3
"""
Test script to run each free model on a small set of puzzles.
This helps verify that each model works correctly with the chess framework.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from main import load_environment, evaluate_puzzles, read_chess_puzzles_csv
from model_interface import ChessModelInterface
import pandas as pd

# Working free models from smoke test
WORKING_FREE_MODELS = [
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "qwen/qwen3-4b:free",  # Reasoning model
]


def test_model(model_name: str, num_puzzles: int = 3, csv_file: str = None, base_url: str = None, delay: float = 0.5):
    """Test a single model on puzzles."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}\n")
    
    # Load environment
    load_environment()
    
    # Get API key and base URL - same way as smoke test
    api_key = os.getenv("ANANNAS_API_KEY")
    if not api_key:
        print("Error: ANANNAS_API_KEY not found in .env file")
        print("Please ensure ANANNAS_API_KEY is set in chess_puzzles/.env")
        return None
    
    if base_url is None:
        base_url = os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
    
    print(f"<debug> : API key loaded: {bool(api_key)}")
    print(f"<debug> : Base URL: {base_url}")
    
    # Default CSV file
    if csv_file is None:
        csv_file = os.path.join(parent_dir, "data", "lichess_puzzles_with_pgn_1000.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return None
    
    # Load puzzles
    try:
        df = read_chess_puzzles_csv(csv_file)
        print(f"Loaded {len(df)} puzzles from {csv_file}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Create model interface
    try:
        model_interface = ChessModelInterface(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            max_completion_tokens=640,  # Will auto-increase for reasoning models
            default_temperature=0.1,
            retry_attempts=2,
        )
    except Exception as e:
        print(f"Error creating model interface: {e}")
        return None
    
    # Test with single model
    output_file = os.path.join(
        parent_dir, 
        "data", 
        f"test_results_{model_name.replace('/', '_').replace(':', '_')}.csv"
    )
    
    # Run evaluation
    try:
        result_df = evaluate_puzzles(
            df=df,
            model_interface=model_interface,
            debate=None,
            debate_v2=None,
            max_puzzles=num_puzzles,
            start_puzzle=0,
            planning_plies=0,
            api_delay=delay,
        )
        
        # Save results
        result_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print summary
        if len(result_df) > 0:
            solved = result_df['puzzle_solved'].sum() if 'puzzle_solved' in result_df.columns else 0
            total = len(result_df)
            print(f"Solved: {solved}/{total} ({100*solved/total:.1f}%)")
            
            # Show token usage if available
            if 'single_model_total_prompt_tokens' in result_df.columns:
                total_prompt = result_df['single_model_total_prompt_tokens'].sum()
                total_completion = result_df['single_model_total_completion_tokens'].sum()
                print(f"Total tokens - Prompt: {total_prompt}, Completion: {total_completion}")
        
        return result_df
    except Exception as e:
        print(f"❌ Error testing model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test free models on chess puzzles"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to test (default: all). Use 'all' to test all models, or specify a model name."
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=50,
        help="Number of puzzles to test per model (default: 50)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between models in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to puzzle CSV file (default: data/lichess_puzzles_with_pgn_1000.csv)"
    )
    parser.add_argument(
        "--anannas-base-url",
        type=str,
        default=None,
        help="Anannas API base URL (defaults to https://api.anannas.ai/v1)"
    )
    
    args = parser.parse_args()
    
    # Determine which models to test
    if args.model.lower() == "all":
        models_to_test = WORKING_FREE_MODELS
    else:
        if args.model not in WORKING_FREE_MODELS:
            print(f"⚠️  Warning: {args.model} not in known working models list")
            print(f"   Known models: {', '.join(WORKING_FREE_MODELS)}")
        models_to_test = [args.model]
    
    print(f"\nTesting {len(models_to_test)} model(s) on {args.num_puzzles} puzzle(s) each")
    print("Using Anannas API for free models")
    print(f"Delay between models: {args.delay}s, Delay between API calls: {args.api_delay}s\n")
    
    results = {}
    api_issues = []
    
    for i, model in enumerate(models_to_test):
        # Add delay between models (except for the first one)
        if i > 0:
            print(f"\nWaiting {args.delay} seconds before next model...")
            time.sleep(args.delay)
        
        result = test_model(model, args.num_puzzles, args.csv_file, args.anannas_base_url, delay=args.api_delay)
        results[model] = result
        
        # Track API issues
        if result is None:
            api_issues.append({
                "model": model,
                "issue": "Failed to initialize or run",
            })
        elif len(result) == 0:
            api_issues.append({
                "model": model,
                "issue": "No results returned",
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = []
    failed = []
    
    for model, result in results.items():
        if result is not None and len(result) > 0:
            solved = result['puzzle_solved'].sum() if 'puzzle_solved' in result.columns else 0
            total = len(result)
            status = "✅" if solved > 0 else "⚠️"
            print(f"{status} {model}: {solved}/{total} solved ({100*solved/total:.1f}%)")
            successful.append(model)
        else:
            print(f"❌ {model}: Failed")
            failed.append(model)
            if model not in [issue["model"] for issue in api_issues]:
                api_issues.append({
                    "model": model,
                    "issue": "Evaluation failed",
                })
    
    # Print API issues summary
    if api_issues:
        print(f"\n{'='*60}")
        print("API ISSUES DETECTED")
        print(f"{'='*60}")
        for issue in api_issues:
            print(f"  ❌ {issue['model']}: {issue.get('issue', 'Unknown error')}")
    
    print(f"\nCompleted: {len(successful)} successful, {len(failed)} failed out of {len(results)} total")


if __name__ == "__main__":
    main()

