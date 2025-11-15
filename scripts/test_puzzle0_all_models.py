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

# Models to test - using Anannas API for all (NOT including OpenAI models)
# Free models + paid models via Anannas
MODELS_TO_TEST = [
    # Free models (high priority)
    # "google/gemini-2.0-flash-exp:free",
    # "together/deepseek-ai-DeepSeek-R1-Distill-Llama-70B-free",
    # "microsoft/phi-3-mini-128k-instruct:free",
    # "qwen/qwen3-4b:free",
    # "mistralai/mistral-small-3.2-24b-instruct:free",
    # "qwen/qwen2.5-vl-72b-instruct:free",
    # "minimax/minimax-m2:free",
    # "google/gemma-3-12b-it:free",
    # "google/gemma-2-9b-it:free",
    # "google/gemma-3-4b-it:free",
    # "together/meta-llama-Llama-3.3-70B-Instruct-Turbo-Free",
    # "qwen/qwen2.5-vl-32b-instruct:free",
    # "qwen/qwen-2.5-coder-32b-instruct:free",
    # "meta-llama/llama-3.3-8b-instruct:free",
    # "google/gemma-3-27b-it:free",
    # "huggingfaceh4/zephyr-7b-beta:free",
    # "qwen/qwen3-8b:free",
    # "liquid/lfm-40b:free",
    
    # # Anthropic via Anannas
    # "anthropic/claude-3.7-sonnet",
    # "anthropic/claude-opus-4.1",
    # "anthropic/claude-3.5-sonnet",
    
    # Qwen models (including the 235b)
    # "qwen/qwen3-max",
    "qwen/qwen3-235b-a22b-instruct-2507",  # The 235b model user mentioned
    # "qwen/qwen2.5-72b-instruct",
    # "qwen/qwen3-30b-a3b-instruct-2507",
    
    # Strong 70B+ models
    # "meta-llama/llama-3.1-70b-instruct",
    # "meta-llama/llama-3.3-70b-instruct",
    # "nvidia/llama-3.1-nemotron-70b-instruct",
    
    # # Other strong models
    # "deepseek/deepseek-v3",
    # "ai21/jamba-instruct",
    # "mistralai/mistral-7b-instruct-v0.3",
]


def _get_api_config_for_model(model_name: str) -> tuple:
    """Determine which API to use based on model name. Returns (api_key, base_url, provider_name)."""
    # Use Anannas API for ALL models (NOT including OpenAI models)
    # Match the exact approach from test_free_models.py
    api_key = os.getenv("ANANNAS_API_KEY")
    base_url = os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
    provider = "Anannas"
    
    return api_key, base_url, provider


def test_model_on_puzzle0(model_name: str) -> dict:
    """Test a single model on puzzle 0 only. Returns result dict."""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    # Load environment (match test_free_models.py approach)
    load_environment()
    
    # Get appropriate API config for this model
    api_key, base_url, provider = _get_api_config_for_model(model_name)
    
    if not api_key:
        error_msg = f"No API key found for {provider}. Please set {'OPENAI_API_KEY' if provider == 'OpenAI' else 'ANANNAS_API_KEY'} in .env"
        print(f"❌ ERROR: {error_msg}")
        return {
            "model": model_name,
            "solved": False,
            "error": error_msg,
        }
    
    print(f"Using {provider} API")
    print(f"<debug> : API key loaded: {bool(api_key)} (provider={provider.lower()})")
    print(f"<debug> : Base URL: {base_url}")
    
    try:
        # Create model interface - match exactly how test_free_models.py does it
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
        error_str = str(e)
        # Check if it's an "invalid model ID" error - model doesn't exist
        if "invalid model" in error_str.lower() or "model ID" in error_str.lower():
            print(f"⚠️  Model not available: {error_str}")
            return {
                "model": model_name,
                "solved": False,
                "error": f"Model not available: {error_str}",
                "correct_moves": 0,
                "total_moves": 0,
            }
        # Check if it's a "no providers available" error - model temporarily unavailable
        if "no providers available" in error_str.lower():
            print(f"⚠️  Model temporarily unavailable via Anannas: {error_str}")
            return {
                "model": model_name,
                "solved": False,
                "error": f"Model temporarily unavailable: {error_str}",
                "correct_moves": 0,
                "total_moves": 0,
            }
        print(f"❌ ERROR: {e}")
        return {
            "model": model_name,
            "solved": False,
            "error": str(e),
            "correct_moves": 0,
            "total_moves": 0,
        }


def main():
    """Test all models on puzzle 0."""
    load_environment()
    
    # Check for Anannas API key (we use Anannas for all models)
    has_anannas = bool(os.getenv("ANANNAS_API_KEY"))
    
    if not has_anannas:
        print("Error: ANANNAS_API_KEY not found in .env")
        print("Please set ANANNAS_API_KEY in chess_puzzles/.env")
        sys.exit(1)
    
    print("="*80)
    print("TESTING ALL MODELS ON PUZZLE 0")
    print("="*80)
    print(f"Total models to test: {len(MODELS_TO_TEST)}")
    print(f"Using Anannas API for all models (NOT including OpenAI models)")
    print(f"Anannas API key: {'✅ Available' if has_anannas else '❌ Not found'}")
    print("="*80)
    
    results = []
    solved_models = []
    
    for i, model in enumerate(MODELS_TO_TEST, 1):
        print(f"\n[{i}/{len(MODELS_TO_TEST)}]")
        result = test_model_on_puzzle0(model)
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

