#!/usr/bin/env python3
"""
Simple test script to verify each free model works with a minimal chess prompt.
This helps isolate issues before running full puzzle evaluations.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dotenv import load_dotenv
from openai import OpenAI

# Load environment
env_file = os.path.join(parent_dir, '.env')
if os.path.exists(env_file):
    load_dotenv(env_file)
    print(f"Loaded environment from {env_file}")
else:
    load_dotenv()

# Working free models
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


def test_model_simple(model_name: str):
    """Test a model with a simple chess prompt."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    api_key = os.getenv("ANANNAS_API_KEY")
    if not api_key:
        print("❌ ANANNAS_API_KEY not found")
        return False
    
    base_url = os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Simple chess prompt (minimal game)
    system_prompt = "You are a chess grandmaster. Provide the next move in standard algebraic notation."
    user_prompt = "1. e4 e5 2. Nf3"
    
    # Check if reasoning model
    is_reasoning = "qwen" in model_name.lower() and "qwen3" in model_name.lower()
    max_tokens = 512 if is_reasoning else 128
    
    try:
        print(f"  Sending request (max_tokens={max_tokens})...")
        # Try with system message first
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            print(f"  ✅ System message supported")
        except Exception as e:
            if "400" in str(e):
                print(f"  ⚠️  System message failed, trying without system message...")
                # Try without system message (combine into user message)
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": combined_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                print(f"  ✅ Works without system message")
            else:
                raise
        
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, 'reasoning', None)
        
        if not content and reasoning:
            content = str(reasoning)[:200]  # Preview reasoning
            print(f"  ✅ Response (from reasoning): {content}...")
        elif content:
            print(f"  ✅ Response: {content[:200]}")
        else:
            print(f"  ⚠️  Empty response (finish_reason: {response.choices[0].finish_reason})")
        
        # Check for chess move patterns
        if any(move in content.upper() for move in ["N", "B", "R", "Q", "K", "E", "D", "C", "F", "G", "H"]):
            print(f"  ✅ Contains potential chess notation")
        else:
            print(f"  ⚠️  No obvious chess notation found")
        
        return True
        
    except Exception as e:
        error_str = str(e)
        if "400" in error_str:
            print(f"  ❌ 400 Error: {error_str[:200]}")
        elif "404" in error_str:
            print(f"  ❌ 404 Error: Model not found")
        else:
            print(f"  ❌ Error: {error_str[:200]}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple test of free models")
    parser.add_argument("--model", type=str, default="all", help="Model to test (default: all)")
    
    args = parser.parse_args()
    
    if args.model.lower() == "all":
        models = WORKING_FREE_MODELS
    else:
        models = [args.model]
    
    print(f"\nTesting {len(models)} model(s) with simple chess prompt\n")
    
    results = {}
    for model in models:
        results[model] = test_model_simple(model)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model}")
    
    successful = sum(results.values())
    print(f"\n{successful}/{len(models)} models working")


if __name__ == "__main__":
    main()

