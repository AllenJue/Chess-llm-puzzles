#!/usr/bin/env python3
"""
Test all specified models on puzzle 0 only.
Does not save results - just shows which models can solve it.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from main import load_environment, evaluate_puzzles, read_chess_puzzles_csv
from model_interface import ChessModelInterface

# Models from the user's list (high-performance models most likely to solve)
MODELS_TO_TEST = [
    # Reasoning models
    "openai/o3-pro",
    "openai/o3-mini",
    "openai/o1-pro",
    "openai/o1-mini",
    
    # Top flagship
    "openai/gpt-5-pro",
    "openai/gpt-5-codex",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-opus-4.1",
    "qwen/qwen3-max",
    "qwen/qwen3-235b-a22b-instruct-2507",  # Already tested - got it right
    
    # Strong 70B+ models
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "qwen/qwen2.5-72b-instruct",
    
    # Fast/strong
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-v3",
    
    # Other strong models
    "ai21/jamba-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "qwen/qwen3-30b-a3b-instruct-2507",
]


def test_model_on_puzzle0(model_name: str, api_key: str, base_url: str) -> dict:
    """Test a single model on puzzle 0 only. Returns result dict."""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Create model interface
        model_interface = ChessModelInterface(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            max_completion_tokens=640,
            default_temperature=0.1,
            retry_attempts=2,
        )
        
        # Load puzzles
        csv_file = os.path.join(parent_dir, "data", "input", "lichess_puzzles_with_pgn_1000.csv")
        df = read_chess_puzzles_csv(csv_file)
        
        # Evaluate only puzzle 0
        df_results = evaluate_puzzles(
            df,
            model_interface=model_interface,
            max_puzzles=1,
            start_puzzle=0,
            planning_plies=0,
            api_delay=0.5,
        )
        
        # Get result
        result = {
            "model": model_name,
            "solved": bool(df_results.iloc[0]["puzzle_solved"]),
            "correct_moves": int(df_results.iloc[0]["correct_moves"]),
            "total_moves": int(df_results.iloc[0]["total_moves"]),
            "error": str(df_results.iloc[0]["error"]),
            "predicted_move": str(df_results.iloc[0]["single_model_move"]),
            "expected_moves": str(df_results.iloc[0]["Moves"]),
        }
        
        if result["solved"]:
            print(f"✅ SOLVED! Correct moves: {result['correct_moves']}/{result['total_moves']}")
        else:
            print(f"❌ Not solved. Correct moves: {result['correct_moves']}/{result['total_moves']}")
            if result["error"]:
                print(f"   Error: {result['error']}")
            print(f"   Predicted: {result['predicted_move']}")
            print(f"   Expected: {result['expected_moves']}")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {
            "model": model_name,
            "solved": False,
            "error": str(e),
        }


def main():
    """Test all models on puzzle 0."""
    load_environment()
    
    # Get API key - try OpenAI first, then Anannas
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANANNAS_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
    
    if not api_key:
        print("Error: No API key found. Please set OPENAI_API_KEY or ANANNAS_API_KEY in .env")
        sys.exit(1)
    
    print("="*80)
    print("TESTING ALL MODELS ON PUZZLE 0")
    print("="*80)
    print(f"Total models to test: {len(MODELS_TO_TEST)}")
    print(f"API Provider: {'OpenAI' if os.getenv('OPENAI_API_KEY') else 'Anannas'}")
    print("="*80)
    
    results = []
    solved_models = []
    
    for i, model in enumerate(MODELS_TO_TEST, 1):
        print(f"\n[{i}/{len(MODELS_TO_TEST)}]")
        result = test_model_on_puzzle0(model, api_key, base_url)
        results.append(result)
        
        if result.get("solved"):
            solved_models.append(model)
        
        # Small delay between models
        if i < len(MODELS_TO_TEST):
            time.sleep(1)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total models tested: {len(results)}")
    print(f"Models that solved puzzle 0: {len(solved_models)}")
    
    if solved_models:
        print("\n✅ SOLVED PUZZLE 0:")
        for model in solved_models:
            print(f"   - {model}")
    else:
        print("\n❌ No models solved puzzle 0")
    
    print("\n" + "="*80)
    print("All results (sorted by correct moves):")
    print("="*80)
    sorted_results = sorted(results, key=lambda x: x.get("correct_moves", 0), reverse=True)
    for r in sorted_results:
        status = "✅" if r.get("solved") else "❌"
        moves = f"{r.get('correct_moves', 0)}/{r.get('total_moves', 0)}"
        print(f"{status} {r['model']:50s} - {moves} correct moves")


if __name__ == "__main__":
    main()

