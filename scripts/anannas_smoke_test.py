import argparse
import os
import re
from typing import List

from openai import OpenAI

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in parent directory (chess_puzzles)
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
except ImportError:
    pass  # dotenv not available, continue without it


# Working free models (tested and confirmed working)
# Note: Some models appear in API /models endpoint but return 404 when called
# (endpoints not actually available). Only models that actually work are listed here.
WORKING_FREE_MODELS: List[str] = [
    # Regular models (no reasoning, content in message.content)
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    # Qwen models that work (some return content directly, some use reasoning)
    "qwen/qwen-2.5-coder-32b-instruct:free",  # Returns content directly
    "qwen/qwen2.5-vl-32b-instruct:free",  # Returns content directly
    "qwen/qwen3-4b:free",  # Uses reasoning mode (content empty, reasoning has text)
    # Models that appear in API but return 404 (endpoints not available):
    # "qwen/qwen2.5-vl-72b-instruct:free" - 404: No endpoints found
    # "qwen/qwen3-8b:free" - 404: No endpoints found
    # "google/gemma-2-9b-it:free" - 404: No endpoints found
    # "microsoft/phi-3-mini-128k-instruct:free" - 404: No endpoints found
    # "minimax/minimax-m2:free" - 404: No endpoints found
    # "liquid/lfm-40b:free" - 404: No endpoints found
    # "huggingfaceh4/zephyr-7b-beta:free" - 404: No endpoints found
]

def get_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def run_smoke_test(client: OpenAI, model: str, max_retries: int = 2) -> None:
    print(f"\n=== Testing model: {model} ===")
    import time
    
    for attempt in range(max_retries + 1):
        try:
            # Create request - don't try to disable reasoning for Qwen models
            # as the API doesn't accept boolean reasoning parameter
            # Reasoning models need more tokens since they show their thinking process
            is_reasoning_model = "qwen" in model.lower() and "qwen3" in model.lower()
            max_tokens_value = 512 if is_reasoning_model else 128
            
            create_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": "Briefly say hello from Anannas."}],
                "max_tokens": max_tokens_value,
                "temperature": 0.3,
            }
            
            completion = client.chat.completions.create(**create_kwargs)
            message = completion.choices[0].message
            reply = message.content
            finish_reason = completion.choices[0].finish_reason if completion.choices else None
            
            # Check for reasoning content (some models use reasoning mode)
            reasoning = getattr(message, 'reasoning', None)
            
            # Check if response is empty or None
            if reply is None or reply.strip() == "":
                if reasoning:
                    # Model used reasoning mode - try to extract final answer from reasoning
                    reasoning_text = reasoning if isinstance(reasoning, str) else str(reasoning)
                    
                    # Try to find a final answer pattern in reasoning
                    # Look for common patterns like "Final answer:", "Answer:", etc.
                    final_answer = None
                    
                    # First, try explicit answer patterns
                    answer_patterns = [
                        "Final answer:",
                        "Answer:",
                        "The answer is",
                        "Hello from Anannas",
                        "So the answer is",
                        "Therefore",
                        "In conclusion",
                    ]
                    
                    for pattern in answer_patterns:
                        if pattern.lower() in reasoning_text.lower():
                            # Extract text after the pattern
                            idx = reasoning_text.lower().find(pattern.lower())
                            if idx != -1:
                                potential_answer = reasoning_text[idx + len(pattern):].strip()
                                # Remove any remaining reasoning markers
                                potential_answer = re.sub(r'<think>', '', potential_answer, flags=re.I)
                                # Take first sentence or up to 200 chars
                                sentences = potential_answer.split('.')
                                if sentences:
                                    final_answer = sentences[0].strip()
                                    if final_answer and len(final_answer) > 3:
                                        break
                    
                    # If no explicit pattern found, try to extract from the end of reasoning
                    # (reasoning models often put the answer at the end)
                    if not final_answer and finish_reason == "length":
                        # Look at the last part of reasoning for a potential answer
                        last_part = reasoning_text[-200:].strip()
                        # Try to find a greeting or answer-like text at the end
                        # Look for sentences that might be the answer
                        sentences = re.split(r'[.!?]\s+', last_part)
                        for sent in reversed(sentences):
                            sent = sent.strip()
                            if len(sent) > 10 and any(word in sent.lower() for word in ["hello", "hi", "greeting", "answer"]):
                                final_answer = sent
                                break
                    
                    if final_answer:
                        print(f"✅ Success (from reasoning): {final_answer}")
                    else:
                        # Show more of the reasoning to help debug
                        preview_len = 300 if finish_reason == "length" else 150
                        print(f"⚠️  Warning: Model returned empty content but has reasoning")
                        print(f"   Finish reason: {finish_reason}")
                        if finish_reason == "length":
                            print(f"   Note: Hit token limit - may need more tokens for reasoning models")
                        print(f"   Reasoning preview: {reasoning_text[:preview_len]}...")
                        print(f"✅ API call succeeded (model used reasoning mode, no clear final answer extracted)")
                else:
                    print(f"⚠️  Warning: Model returned empty response")
                    print(f"   Finish reason: {completion.choices[0].finish_reason}")
                    print(f"✅ API call succeeded (but response is empty)")
            else:
                print(f"✅ Success: {reply}")
            return
        except Exception as exc:  # noqa: BLE001
            error_str = str(exc)
            # Check if it's a transient error (503, 429, etc.)
            is_transient = any(code in error_str for code in ["503", "429", "temporarily unavailable", "rate limit"])
            
            if attempt < max_retries and is_transient:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s
                print(f"⚠️  Transient error (attempt {attempt + 1}/{max_retries + 1}): {exc}")
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed: {exc}")
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke test against the Anannas API.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Models to test (defaults to all available free models).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1"),
        help="Override the Anannas base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ANANNAS_API_KEY"),
        help="Override the Anannas API key (falls back to ANANNAS_API_KEY env).",
    )

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("ANANNAS_API_KEY not provided. Set the env var or use --api-key.")

    # Determine which models to test (default to working free models)
    if args.models:
        models = args.models
    else:
        models = WORKING_FREE_MODELS
        print(f"Testing {len(models)} working free models:")
        for model in models:
            print(f"  - {model}")
    
    client = get_client(api_key=args.api_key, base_url=args.base_url)

    for model in models:
        run_smoke_test(client, model)


if __name__ == "__main__":
    main()

