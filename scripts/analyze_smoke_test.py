#!/usr/bin/env python3
"""
Analyze smoke test results and organize models by family and cost.
"""

import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Pricing information extracted from Anannas API
# Format: model_name: (input_cost_per_1M, output_cost_per_1M, context_length, tier)
PRICING_DATA: Dict[str, Tuple[float, float, int, str]] = {
    # Free models
    "google/gemini-2.0-flash-exp:free": (0.00, 0.00, 1000000, "light"),
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo-Free": (0.00, 0.00, 131000, "light"),
    "qwen/qwen2.5-vl-32b-instruct:free": (0.00, 0.00, 16000, "standard"),
    "qwen/qwen-2.5-coder-32b-instruct:free": (0.00, 0.00, 33000, "standard"),
    "meta-llama/llama-3.3-8b-instruct:free": (0.00, 0.00, 128000, "light"),
    "google/gemma-3-27b-it:free": (0.00, 0.00, 96000, "fast"),
    "huggingfaceh4/zephyr-7b-beta:free": (0.00, 0.00, 4000, "light"),
    "qwen/qwen3-8b:free": (0.00, 0.00, 41000, "light"),
    "liquid/lfm-40b:free": (0.00, 0.00, 33000, "fast"),
    "together/togethercomputer-MoA-1-Turbo": (0.00, 0.00, 33000, "light"),
    "together/togethercomputer-MoA-1": (0.00, 0.00, 33000, "light"),
    "microsoft/phi-4-reasoning": (0.00, 0.00, 33000, "reasoning"),
    "nousresearch/hermes-2-theta-llama-3-8b": (0.00, 0.00, 16000, "reasoning"),
    "together/meta-llama-Meta-Llama-3.1-405B-Instruct-Lite-Pro": (0.00, 0.00, 4000, "light"),
    "together/ServiceNow-AI-Apriel-1.5-15b-Thinker": (0.00, 0.00, 131000, "light"),
    "mistralai/mistral-small-3.2-24b-instruct:free": (0.00, 0.00, 131000, "light"),
    "qwen/qwen2.5-vl-72b-instruct:free": (0.00, 0.00, 131000, "standard"),
    "minimax/minimax-m2:free": (0.00, 0.00, 205000, "light"),
    "google/gemma-3-12b-it:free": (0.00, 0.00, 33000, "light"),
    "google/gemma-2-9b-it:free": (0.00, 0.00, 8000, "light"),
    "google/gemma-3-4b-it:free": (0.00, 0.00, 33000, "light"),
    
    # OpenAI models
    "openai/gpt-oss-20b": (0.04, 0.15, 131000, "light"),
    "openai/gpt-5-codex": (5.00, 40.00, 400000, "flagship"),
    "openai/gpt-4o-mini-search-preview": (0.15, 0.60, 128000, "fast"),
    "openai/gpt-4.1": (2.00, 8.00, 1000000, "flagship"),
    "openai/gpt-5-chat": (1.25, 10.00, 128000, "flagship"),
    "openai/gpt-4-turbo": (10.00, 30.00, 128000, "fast"),
    "openai/gpt-4o-audio-preview": (2.50, 10.00, 128000, "flagship"),
    "openai/codex-mini": (1.50, 6.00, 200000, "fast"),
    "openai/gpt-5.1-codex": (1.25, 10.00, 400000, "flagship"),
    "openai/gpt-4o-2024-08-06": (2.50, 10.00, 128000, "fast"),
    "openai/gpt-5.1": (1.25, 10.00, 400000, "flagship"),
    "openai/gpt-4o": (2.50, 10.00, 128000, "fast"),
    "openai/gpt-5": (1.25, 10.00, 400000, "flagship"),
    "openai/gpt-4.1-nano": (0.10, 0.40, 1000000, "light"),
    "openai/gpt-5.1-codex-mini": (0.25, 2.00, 400000, "flagship"),
    "openai/gpt-4o-2024-05-13": (5.00, 15.00, 128000, "fast"),
    "openai/gpt-4o-2024-11-20": (2.50, 10.00, 128000, "fast"),
    "openai/gpt-5-mini": (0.25, 2.00, 400000, "fast"),
    "openai/gpt-4o:extended": (6.00, 18.00, 128000, "fast"),
    "openai/chatgpt-4o-latest": (5.00, 15.00, 128000, "fast"),
    "openai/gpt-4.1-mini": (0.40, 1.60, 1000000, "fast"),
    "openai/gpt-4o-mini-2024-07-18": (0.15, 0.60, 128000, "fast"),
    "openai/gpt-4o-mini": (0.15, 0.60, 128000, "fast"),
    "openai/gpt-5-nano": (0.05, 0.40, 400000, "light"),
    "openai/gpt-5-pro": (10.00, 100.00, 400000, "flagship"),
    "openai/gpt-5.1-chat": (1.25, 10.00, 128000, "flagship"),
    "openai/gpt-oss-120b": (0.07, 0.28, 131000, "fast"),
    "openai/gpt-4o-search-preview": (2.50, 10.00, 128000, "fast"),
    "openai/o1": (15.00, 60.00, 200000, "reasoning"),
    "openai/o4-mini": (1.10, 4.40, 200000, "reasoning"),
    "openai/o3-mini": (1.10, 4.40, 200000, "reasoning"),
    "openai/o3": (2.00, 8.00, 200000, "reasoning"),
    "openai/o1-pro": (150.00, 600.00, 200000, "reasoning"),
    "openai/o3-pro": (20.00, 80.00, 200000, "reasoning"),
    "openai/o1-mini-2024-09-12": (1.10, 4.40, 128000, "reasoning"),
    "openai/o1-mini": (1.10, 4.40, 128000, "reasoning"),
    
    # Anthropic models
    "anthropic/claude-3.7-sonnet": (3.00, 15.00, 200000, "fast"),
    "anthropic/claude-haiku-4.5": (1.00, 5.00, 200000, "light"),
    "anthropic/claude-sonnet-4": (3.00, 15.00, 200000, "fast"),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00, 200000, "fast"),
    "anthropic/claude-opus-4": (15.00, 75.00, 200000, "flagship"),
    "anthropic/claude-3-sonnet": (3.00, 15.00, 200000, "fast"),
    "anthropic/claude-3-haiku": (0.25, 1.25, 200000, "light"),
    "anthropic/claude-3-opus": (15.00, 75.00, 200000, "flagship"),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00, 200000, "fast"),
    "anthropic/claude-opus-4.1": (15.00, 75.00, 200000, "flagship"),
    "anthropic/claude-3.5-haiku": (0.80, 4.00, 200000, "light"),
    
    # Qwen models
    "qwen/qwen3-next-80b-a3b-thinking": (0.15, 1.20, 262000, "reasoning"),
    "qwen/qwen3-next-80b-a3b-instruct": (0.10, 0.80, 262000, "standard"),
    "qwen/qwen3-vl-30b-a3b-thinking": (0.20, 1.00, 262000, "reasoning"),
    "together/Qwen-Qwen3-235B-A22B-Instruct-2507-tput": (0.20, 0.60, 262000, "flagship"),
    "qwen/qwen-2.5-coder-32b-instruct:free": (0.00, 0.00, 33000, "standard"),
    "qwen/qwen2.5-72b-instruct": (0.13, 0.40, 128000, "fast"),
    "qwen/qwen3-coder": (0.38, 1.53, 262000, "standard"),
    "qwen/qwen3-235b-a22b-thinking-2507": (0.20, 0.80, 262000, "reasoning"),
    "qwen/qwen2.5-coder-7b": (0.03, 0.09, 32000, "light"),
    "qwen/qwq-32b-preview": (0.18, 0.18, 33000, "reasoning"),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.22, 0.88, 131000, "standard"),
    "qwen/qwen-2.5-72b-instruct": (0.36, 0.36, 33000, "fast"),
    "qwen/qwen3-coder-plus": (1.00, 5.00, 128000, "standard"),
    "qwen/qwen3-30b-a3b": (0.10, 0.30, 41000, "fast"),
    "qwen/qwen3-coder-480b-a35b-instruct": (0.40, 1.80, 262000, "fast"),
    "qwen/qwen3-vl-8b-thinking": (0.18, 2.10, 256000, "reasoning"),
    "qwen/qwen3-235b-a22b-instruct-2507": (0.20, 0.60, 262000, "fast"),
    "qwen/qwen3-235b-a22b": (0.18, 0.54, 131000, "standard"),
    "qwen/qwen3-vl-8b-instruct": (0.08, 0.50, 262000, "light"),
    "qwen/qwen3-235b-a22b-2507": (0.08, 0.55, 262000, "standard"),
    "qwen/qwen3-30b-a3b-instruct-2507": (0.10, 0.30, 262000, "fast"),
    "qwen/qwen3-coder-30b-a3b-instruct": (0.10, 0.30, 262000, "fast"),
    "qwen/qwen3-30b-a3b-thinking-2507": (0.10, 0.30, 262000, "reasoning"),
    "qwen/qwen3-max": (1.20, 6.00, 256000, "standard"),
    "qwen/qwen-turbo": (0.05, 0.20, 1000000, "fast"),
    "qwen/qwen-plus": (0.40, 1.20, 131000, "standard"),
    "qwen/qwen-max": (1.60, 6.40, 33000, "standard"),
    "qwen/qwen-2.5-vl-7b-instruct": (0.20, 0.20, 33000, "light"),
    "qwen/qwen2.5-vl-72b-instruct": (0.08, 0.33, 33000, "standard"),
    "qwen/qwen-2.5-coder-32b-instruct": (0.04, 0.16, 33000, "standard"),
    "qwen/qwen-2.5-7b-instruct": (0.18, 0.18, 33000, "light"),
    "qwen/qwen2.5-coder-7b-instruct": (0.03, 0.09, 33000, "light"),
    "qwen/qwen-vl-plus": (0.21, 0.63, 8000, "light"),
    "qwen/qwen-vl-max": (0.80, 3.20, 131000, "standard"),
    "qwen/qwen-plus-2025-07-28": (0.40, 4.00, 1000000, "standard"),
    "qwen/qwq-32b": (0.15, 0.45, 32000, "reasoning"),
    "qwen/qwen3-14b": (0.08, 0.24, 41000, "light"),
    "qwen/qwen3-32b": (0.10, 0.30, 41000, "fast"),
    "qwen/qwen3-4b:free": (0.00, 0.00, 41000, "light"),
    "qwen/qwen3-8b:free": (0.00, 0.00, 41000, "light"),
    "qwen/qwen3-coder-flash": (0.30, 1.50, 128000, "fast"),
    
    # Meta Llama models
    "meta-llama/llama-3.2-3b-instruct": (0.02, 0.02, 131000, "light"),
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88, 131000, "flagship"),
    "meta-llama/llama-3.3-8b-instruct:free": (0.00, 0.00, 128000, "light"),
    "meta-llama/llama-3.1-405b-instruct": (3.00, 3.00, 33000, "flagship"),
    "meta-llama/llama-3.1-8b-instruct": (0.18, 0.18, 131000, "light"),
    "meta-llama/llama-3.2-90b-vision-instruct": (0.35, 0.40, 33000, "standard"),
    "meta-llama/llama-3.3-70b-instruct": (0.88, 0.88, 131000, "fast"),
    "meta-llama/llama-3-70b-instruct": (0.30, 0.40, 8000, "standard"),
    "meta-llama/llama-3-8b-instruct": (0.03, 0.06, 8000, "light"),
    "meta-llama/llama-4-maverick": (0.15, 0.60, 1000000, "standard"),
    "meta-llama/llama-guard-4-12b": (0.18, 0.18, 164000, "light"),
    "meta-llama/llama-guard-3-8b": (0.02, 0.06, 131000, "light"),
    "meta-llama/llama-3.2-11b-vision-instruct": (0.05, 0.05, 131000, "standard"),
    "meta-llama/llama-3.1-405b": (4.00, 4.00, 33000, "flagship"),
    "meta-llama/llama-3.1-8b-instruct-fast": (0.03, 0.09, 131000, "fast"),
    "meta-llama/llama-3.3-70b-instruct-fast": (0.13, 0.40, 131000, "fast"),
    "meta-llama/llama-3.1-70b-instruct": (0.88, 0.88, 131000, "fast"),
    "meta-llama/llama-3.2-1b-instruct": (0.01, 0.01, 131000, "light"),
    "meta-llama/llama-4-scout": (0.08, 0.30, 328000, "standard"),
    "meta-llama/llama-guard-2-8b": (0.20, 0.20, 8000, "light"),
    
    # Google models
    "google/gemini-2.0-flash-001": (0.10, 0.40, 1000000, "fast"),
    "google/gemma-3-12b-it": (0.03, 0.14, 96000, "light"),
    "google/gemma-2-9b-it": (0.02, 0.04, 8000, "light"),
    "google/gemini-2.5-flash-image-preview": (0.30, 2.50, 33000, "fast"),
    "google/gemini-flash-1.5-8b": (0.04, 0.15, 1000000, "light"),
    "google/gemini-2.0-flash-lite-001": (0.07, 0.30, 1000000, "light"),
    "google/gemini-2.5-pro-preview": (1.25, 10.00, 1000000, "flagship"),
    "google/gemma-2-27b-it": (0.65, 0.65, 8000, "fast"),
    "google/gemini-2.5-flash-lite-preview-06-17": (0.10, 0.40, 1000000, "light"),
    "google/gemma-3-27b-it": (0.06, 0.26, 96000, "fast"),
    "google/gemini-2.5-pro": (1.25, 10.00, 1000000, "flagship"),
    "google/gemini-flash-1.5": (0.07, 0.30, 1000000, "light"),
    "google/gemini-2.5-flash": (0.30, 2.50, 1000000, "fast"),
    "google/gemini-2.5-pro-preview-05-06": (1.25, 10.00, 1000000, "flagship"),
    "google/gemini-2.5-flash-lite": (0.10, 0.40, 1000000, "light"),
    "google/gemma-2-2b-it": (0.02, 0.06, 8000, "light"),
    "google/gemma-3-4b-it": (0.04, 0.08, 131000, "light"),
    "google/gemini-pro-1.5": (1.25, 5.00, 2000000, "flagship"),
    "google/gemma-2-9b-it-fast": (0.03, 0.09, 8000, "light"),
    
    # Mistral models
    "mistralai/mistral-small-3.2-24b-instruct": (0.05, 0.10, 128000, "light"),
    "mistralai/pixtral-large-2411": (2.00, 6.00, 131000, "standard"),
    "mistralai/mistral-large-2411": (2.00, 6.00, 131000, "standard"),
    "mistralai/mistral-7b-instruct-v0.2": (0.20, 0.20, 33000, "light"),
    "mistralai/ministral-8b": (0.10, 0.10, 128000, "light"),
    "mistralai/mistral-tiny": (0.25, 0.25, 33000, "light"),
    "mistralai/pixtral-12b": (0.10, 0.10, 33000, "light"),
    "mistralai/mistral-medium-3.1": (0.40, 2.00, 131000, "fast"),
    "mistralai/mistral-small": (0.20, 0.60, 33000, "light"),
    "mistralai/mistral-large": (2.00, 6.00, 128000, "standard"),
    "mistralai/mixtral-8x22b-instruct": (2.00, 6.00, 66000, "standard"),
    "mistralai/mistral-7b-instruct": (0.03, 0.05, 33000, "light"),
    "mistralai/mistral-7b-instruct-v0.1": (0.11, 0.19, 3000, "light"),
    "mistralai/mistral-7b-instruct-v0.3": (0.03, 0.05, 33000, "light"),
    "mistralai/mixtral-8x7b-instruct": (0.54, 0.54, 33000, "standard"),
    "mistralai/mistral-saba": (0.20, 0.60, 33000, "light"),
    "mistralai/mistral-nemo": (0.02, 0.04, 131000, "standard"),
    "mistralai/ministral-3b": (0.04, 0.04, 33000, "light"),
    "mistralai/mistral-large-2407": (2.00, 6.00, 131000, "standard"),
    "mistralai/magistral-medium-2506": (2.00, 5.00, 41000, "flagship"),
    "mistralai/magistral-small-2506": (0.50, 1.50, 40000, "fast"),
    "mistralai/devstral-small": (0.07, 0.28, 128000, "light"),
    "mistralai/devstral-medium": (0.40, 2.00, 131000, "fast"),
    "mistralai/devstral-small-2505": (0.08, 0.24, 128000, "light"),
    "mistralai/codestral-2501": (0.30, 0.90, 262000, "standard"),
    "mistralai/codestral-2508": (0.30, 0.90, 256000, "fast"),
    "mistralai/mistral-small-3.1-24b-instruct": (0.05, 0.08, 128000, "light"),
    "mistralai/mistral-small-24b-instruct-2501": (0.05, 0.08, 33000, "light"),
    
    # DeepSeek models
    "deepseek/deepseek-r1-0528": (0.80, 2.40, 164000, "reasoning"),
    "deepseek/deepseek-r1-distill-qwen-32b": (0.27, 0.27, 131000, "standard"),
    "deepseek-ai/deepseek-v3": (0.50, 1.50, 164000, "fast"),
    "deepseek/deepseek-r1-0528-qwen3-8b": (0.03, 0.11, 131000, "reasoning"),
    "deepseek/deepseek-v3.2-exp": (0.27, 0.40, 164000, "standard"),
    "deepseek/deepseek-chat-v3.1": (0.20, 0.80, 164000, "standard"),
    "deepseek/deepseek-prover-v2": (0.50, 2.18, 164000, "reasoning"),
    "deepseek/deepseek-v3": (0.50, 1.50, 128000, "fast"),
    "deepseek/deepseek-v3-0324": (0.50, 1.50, 128000, "fast"),
    "deepseek-ai/deepseek-v3-0324": (0.50, 1.50, 164000, "fast"),
    "deepseek-ai/deepseek-v3-0324-fast": (0.75, 2.25, 33000, "fast"),
    "deepseek/deepseek-chat": (0.14, 0.28, 128000, "fast"),
    "deepseek/deepseek-chat-v3-0324": (0.24, 0.84, 164000, "standard"),
    "deepseek/deepseek-r1": (0.55, 2.16, 128000, "reasoning"),
    "deepseek/deepseek-r1-distill-llama-70b": (0.55, 2.16, 128000, "fast"),
    "deepseek/deepseek-r1-distill-qwen-14b": (0.15, 0.15, 33000, "light"),
    "deepseek/deepseek-v3.1-terminus": (0.27, 1.00, 164000, "standard"),
    
    # Microsoft models
    "microsoft/phi-3-mini-128k-instruct": (0.01, 0.01, 128000, "light"),
    "microsoft/phi-3-mini-128k-instruct:free": (0.00, 0.00, 128000, "light"),
    "microsoft/phi-4-reasoning": (0.00, 0.00, 33000, "reasoning"),
    "microsoft/phi-4": (0.06, 0.14, 16000, "light"),
    "microsoft/phi-3-medium-128k-instruct": (0.01, 0.01, 128000, "light"),
    "microsoft/phi-3.5-mini-128k-instruct": (0.10, 0.10, 128000, "light"),
    "microsoft/phi-4-reasoning-plus": (0.07, 0.35, 33000, "reasoning"),
    "microsoft/phi-4-multimodal-instruct": (0.05, 0.10, 131000, "standard"),
    "microsoft/mai-ds-r1": (0.30, 1.20, 164000, "reasoning"),
    "microsoft/wizardlm-2-8x22b": (0.48, 0.48, 66000, "standard"),
    
    # Other models (add more as needed)
    "arcee-ai/maestro-reasoning": (0.90, 3.30, 131000, "reasoning"),
    "arcee-ai/spotlight": (0.18, 0.18, 131000, "standard"),
    "arcee-ai/coder-large": (0.50, 0.80, 33000, "standard"),
    "arcee-ai/afm-4.5b": (0.01, 0.01, 4000, "light"),
    "arcee-ai/virtuoso-large": (0.75, 1.20, 131000, "standard"),
    "amazon/nova-pro-v1": (0.80, 3.20, 300000, "flagship"),
    "amazon/nova-lite-v1": (0.06, 0.24, 300000, "light"),
    "amazon/nova-micro-v1": (0.03, 0.14, 128000, "light"),
    "cohere/command-r7b-12-2024": (0.04, 0.15, 128000, "light"),
    "cohere/command-r-08-2024": (0.15, 0.60, 128000, "standard"),
    "cohere/command-a": (2.50, 10.00, 256000, "standard"),
    "cohere/command-r": (0.50, 1.50, 128000, "fast"),
    "cohere/command-r-plus-08-2024": (2.50, 10.00, 128000, "standard"),
    "cohere/command-r-plus": (3.00, 15.00, 128000, "flagship"),
    "ai21/jamba-instruct": (0.35, 1.40, 256000, "light"),
    "ai21/jamba-large-1.7": (2.00, 8.00, 256000, "flagship"),
    "nvidia/llama-3.1-nemotron-70b-instruct": (0.60, 0.60, 131000, "standard"),
    "nvidia/llama-3_1-nemotron-ultra-253b-v1": (0.60, 1.80, 128000, "flagship"),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": (0.60, 1.80, 131000, "flagship"),
    "nvidia/nemotron-nano-9b-v2": (0.04, 0.16, 131000, "light"),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (0.10, 0.40, 131000, "standard"),
    "z-ai/glm-4.5-air": (0.20, 1.20, 128000, "fast"),
    "z-ai/glm-4.5": (0.60, 2.20, 128000, "fast"),
    "z-ai/glm-4.6": (0.60, 2.20, 128000, "fast"),
    "bytedance/seed-oss-36b-instruct": (0.35, 0.35, 33000, "fast"),
    "allenai/llama-3.1-tulu-3-405b": (3.00, 3.00, 131000, "flagship"),
    "allenai/molmo-7b-d": (0.07, 0.07, 8000, "light"),
    "allenai/olmo-7b-instruct": (0.10, 0.10, 4000, "light"),
    "allenai/olmo-2-0325-32b-instruct": (0.50, 0.50, 33000, "fast"),
    "grok/grok-2-vision-1212": (2.00, 10.00, 33000, "flagship"),
    "grok/grok-3-mini": (0.30, 0.50, 131000, "standard"),
    "grok/grok-code-fast-1": (0.20, 1.50, 256000, "standard"),
    "grok/grok-3": (3.00, 15.00, 131000, "flagship"),
    "grok/grok-4-fast-non-reasoning": (0.20, 0.50, 2000000, "standard"),
    "grok/grok-4": (3.00, 15.00, 256000, "flagship"),
    "grok/grok-4-fast-reasoning": (0.20, 0.50, 2000000, "standard"),
    "grok/grok-4-0709": (3.00, 15.00, 256000, "flagship"),
    "liquid/lfm-40b": (1.00, 1.00, 33000, "fast"),
    "liquid/lfm-7b": (0.05, 0.05, 4000, "light"),
    "minimax/minimax-m1": (0.40, 2.20, 1000000, "flagship"),
    "minimax/minimax-m2:free": (0.00, 0.00, 205000, "light"),
    "moonshotai/kimi-k2-0711": (0.60, 2.50, 131000, "flagship"),
    "moonshotai/kimi-k2-instruct": (0.50, 2.40, 131000, "fast"),
    "moonshotai/kimi-k2-0905": (0.60, 2.50, 262000, "flagship"),
    "moonshotai/kimi-k2-thinking": (0.60, 2.50, 262000, "reasoning"),
    "perplexity/sonar-deep-research": (2.00, 8.00, 128000, "reasoning"),
    "perplexity/sonar-pro": (3.00, 15.00, 200000, "standard"),
    "perplexity/sonar-reasoning-pro": (2.00, 8.00, 128000, "reasoning"),
    "perplexity/sonar-reasoning": (1.00, 5.00, 127000, "reasoning"),
    "perplexity/llama-3.1-sonar-huge-128k-online": (5.00, 5.00, 127000, "flagship"),
    "together/deepseek-ai-DeepSeek-R1-0528-tput": (0.55, 2.19, 164000, "flagship"),
    "together/Qwen-Qwen2.5-72B-Instruct-Turbo": (1.20, 1.20, 131000, "flagship"),
    "together/meta-llama-Meta-Llama-3.1-8B-Instruct-Reference": (0.20, 0.20, 16000, "fast"),
    "together/meta-llama-Meta-Llama-3-8B-Instruct": (0.20, 0.20, 8000, "fast"),
    "together/meta-llama-Llama-3.2-3B-Instruct-Turbo": (0.06, 0.06, 131000, "light"),
    "together/meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18, 131000, "flagship"),
    "together/meta-llama-Meta-Llama-3-70B-Instruct-Turbo": (0.88, 0.88, 8000, "fast"),
    "together/meta-llama-Meta-Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88, 131000, "flagship"),
    "together/meta-llama-Meta-Llama-3.1-405B-Instruct-Turbo": (3.50, 3.50, 131000, "flagship"),
    "together/meta-llama-Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59, 1000000, "flagship"),
    "together/meta-llama-Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85, 1000000, "flagship"),
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo-batch": (0.88, 0.88, 131000, "flagship"),
    "together/meta-llama-Meta-Llama-3.1-70B-Instruct-Reference": (0.90, 0.90, 8000, "fast"),
    "together/meta-llama-Meta-Llama-3-8B-Instruct-Lite": (0.10, 0.10, 8000, "fast"),
    "together/Qwen-Qwen3-Coder-480B-A35B-Instruct-FP8": (2.00, 2.00, 262000, "flagship"),
    "together/Qwen-Qwen3-235B-A22B-fp8-tput": (0.20, 0.60, 41000, "fast"),
    "together/Qwen-Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30, 33000, "fast"),
    "together/Qwen-Qwen2.5-14B-Instruct": (0.80, 0.80, 33000, "fast"),
    "together/zai-org-GLM-4.5-Air-FP8": (0.20, 1.10, 131000, "flagship"),
    "together/arize-ai-qwen-2-1.5b-instruct": (0.10, 0.10, 33000, "fast"),
    "together/arcee_ai-arcee-spotlight": (0.18, 0.18, 131000, "flagship"),
    "together/deepseek-ai-DeepSeek-V3.1": (0.60, 1.70, 131000, "flagship"),
    "together/deepcogito-cogito-v2-preview-deepseek-671b": (1.25, 1.25, 164000, "flagship"),
    "together/deepcogito-cogito-v2-preview-llama-109B-MoE": (0.18, 0.59, 33000, "fast"),
    "together/deepcogito-cogito-v2-preview-llama-405B": (3.50, 3.50, 33000, "fast"),
    "together/deepcogito-cogito-v2-preview-llama-70B": (0.88, 0.88, 33000, "fast"),
    "together/scb10x-scb10x-typhoon-2-1-gemma3-12b": (0.20, 0.20, 131000, "flagship"),
    "together/google-gemma-3n-E4B-it": (0.02, 0.04, 33000, "light"),
    "together/togethercomputer-Refuel-Llm-V2": (0.60, 0.60, 16000, "fast"),
    "together/togethercomputer-Refuel-Llm-V2-Small": (0.20, 0.20, 8000, "fast"),
    "together/moonshotai-Kimi-K2-Instruct-0905": (1.00, 3.00, 262000, "flagship"),
    "together/nvidia-NVIDIA-Nemotron-Nano-9B-v2": (0.06, 0.25, 131000, "light"),
    "together/marin-community-marin-8b-instruct": (0.18, 0.18, 4000, "fast"),
    "nousresearch/deephermes-3-llama-3-8b-preview": (0.03, 0.11, 131000, "standard"),
    "nousresearch/deephermes-3-mistral-24b-preview": (0.15, 0.59, 33000, "standard"),
    "nousresearch/hermes-4-70b": (0.13, 0.40, 128000, "fast"),
    "nousresearch/hermes-3-llama-3.1-405b": (1.00, 3.00, 128000, "flagship"),
    "nousresearch/hermes-3-llama-3.1-70b": (0.30, 0.30, 66000, "standard"),
    "nousresearch/hermes-2-pro-llama-3-8b": (0.03, 0.08, 33000, "fast"),
    "nousresearch/hermes-4-405b": (1.00, 3.00, 128000, "flagship"),
    "samba-nova/llama-3.2-90b-vision-instruct": (0.35, 0.40, 4000, "fast"),
    "samba-nova/llama-3.2-11b-vision-instruct": (0.04, 0.04, 4000, "light"),
    "samba-nova/meta-llama-3.2-3b-instruct": (0.02, 0.02, 4000, "light"),
    "samba-nova/meta-llama-3.2-1b-instruct": (0.01, 0.01, 4000, "light"),
    "sao10k/l3.1-70b-hanami-x1": (1.50, 1.50, 33000, "fast"),
    "sao10k/l3-euryale-70b": (1.35, 1.35, 8000, "fast"),
    "mattshumer/reflection-70b": (0.31, 0.31, 131000, "fast"),
    "neversleep/llama-3.1-lumimaid-70b": (4.00, 4.00, 33000, "fast"),
    "neversleep/llama-3.1-lumimaid-8b": (0.23, 1.25, 25000, "light"),
    "thedrummer/cydonia-24b-v4.1": (0.75, 0.75, 33000, "fast"),
    "thedrummer/anubis-70b-v1.1": (2.00, 2.00, 8000, "fast"),
    "aetherwiing/mn-starcannon-12b": (0.15, 0.15, 8000, "light"),
    "aion-labs/aion-rp-llama-3.1-8b": (0.10, 0.10, 8000, "light"),
    "aion-labs/aion-1.0": (0.15, 0.15, 8000, "light"),
    "agentica-org/deepcoder-14b-preview": (0.15, 0.15, 16000, "light"),
    "alpindale/goliath-120b": (2.50, 2.50, 8000, "flagship"),
    "alpindale/magnum-72b": (3.75, 4.50, 16000, "fast"),
    "anthracite-org/magnum-v2-72b": (0.75, 0.75, 33000, "fast"),
    "baidu/ernie-4.5-300b-a47b": (0.30, 0.30, 123000, "flagship"),
    "cognitivecomputations/dolphin-mistral-24b-venice-edition": (0.80, 0.80, 33000, "fast"),
    "cognitivecomputations/dolphin-mixtral-8x7b": (0.50, 0.50, 32000, "fast"),
    "eva/eva-qwen-2.5-32b": (0.60, 0.60, 33000, "fast"),
    "eva/eva-qwen-2.5-14b": (0.18, 0.18, 33000, "light"),
    "eva-unit-01/eva-qwen-2.5-14b": (0.20, 0.20, 33000, "light"),
    "inflection/inflection-3-pi": (1.00, 1.00, 8000, "fast"),
    "inflection/inflection-3-productivity": (1.00, 1.00, 8000, "fast"),
    "inclusionai/ring-1t": (0.57, 2.28, 131000, "flagship"),
    "inclusionai/ling-1t": (0.90, 0.90, 131000, "flagship"),
    "jondurbin/airoboros-l2-70b": (0.70, 0.90, 4000, "fast"),
    "koboldai/psyfighter-13b-2": (0.01, 0.01, 4000, "light"),
    "meituan/longcat-flash-chat": (0.02, 0.02, 8000, "light"),
    "pygmalionai/pygmalion-13b-4bit": (0.01, 0.01, 4000, "light"),
    "pygmalionai/mythalion-13b": (0.01, 0.01, 4000, "light"),
    "stepfun-ai/step3": (0.90, 0.90, 131000, "flagship"),
    "tencent/hunyuan-a13b-instruct": (0.07, 0.07, 8000, "light"),
    "alibaba/tongyi-deepresearch-30b-a3b": (0.60, 0.60, 33000, "fast"),
    "all-hands/openhands-lm-32b-v0.1": (0.30, 0.30, 16000, "light"),
    "01-ai/yi-large": (3.00, 3.00, 33000, "fast"),
    "01-ai/yi-large-fc": (0.90, 0.90, 33000, "fast"),
    "01-ai/yi-large-turbo": (0.80, 0.80, 33000, "fast"),
    "01-ai/yi-lightning": (0.03, 0.12, 16000, "light"),
    "01-ai/yi-vision": (0.80, 0.80, 16000, "fast"),
}


def extract_model_family(model_name: str) -> str:
    """Extract model family from model name."""
    # Remove provider prefix if present
    parts = model_name.split("/")
    if len(parts) > 1:
        provider = parts[0]
        model = parts[1]
        
        # Special handling for some providers
        if provider == "together":
            # Extract the actual model family from together models
            if "meta-llama" in model.lower() or "llama" in model.lower():
                return "meta-llama"
            elif "qwen" in model.lower():
                return "qwen"
            elif "deepseek" in model.lower():
                return "deepseek"
            elif "gemma" in model.lower():
                return "google"
            else:
                return provider
        elif provider == "meta-llama":
            return "meta-llama"
        elif provider == "qwen":
            return "qwen"
        elif provider == "openai":
            return "openai"
        elif provider == "anthropic":
            return "anthropic"
        elif provider == "google" or "gemma" in model.lower() or "gemini" in model.lower():
            return "google"
        elif provider == "mistralai":
            return "mistralai"
        elif provider == "deepseek" or provider == "deepseek-ai":
            return "deepseek"
        elif provider == "microsoft":
            return "microsoft"
        elif provider == "nvidia":
            return "nvidia"
        elif provider == "cohere":
            return "cohere"
        elif provider == "ai21":
            return "ai21"
        elif provider == "amazon":
            return "amazon"
        elif provider == "arcee-ai":
            return "arcee-ai"
        elif provider == "grok":
            return "grok"
        elif provider == "perplexity":
            return "perplexity"
        elif provider == "moonshotai":
            return "moonshotai"
        elif provider == "minimax":
            return "minimax"
        elif provider == "liquid":
            return "liquid"
        elif provider == "z-ai":
            return "z-ai"
        elif provider == "nousresearch":
            return "nousresearch"
        elif provider == "samba-nova":
            return "samba-nova"
        else:
            return provider
    return "other"


def get_total_cost_per_1M(input_cost: float, output_cost: float) -> float:
    """Calculate total cost assuming 1:1 input/output ratio."""
    return input_cost + output_cost


def analyze_smoke_test_results(csv_file: str, output_file: Optional[str] = None):
    """Analyze smoke test results and organize by family and cost."""
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Filter to only successful models
    successful = df[df["status"] == "success"].copy()
    
    print(f"Total models tested: {len(df)}")
    print(f"Successful models: {len(successful)}")
    print(f"Failed models: {len(df) - len(successful)}")
    print()
    
    # Add model family and pricing info
    successful["family"] = successful["model"].apply(extract_model_family)
    successful["input_cost"] = successful["model"].apply(
        lambda m: PRICING_DATA.get(m, (None, None, None, None))[0]
    )
    successful["output_cost"] = successful["model"].apply(
        lambda m: PRICING_DATA.get(m, (None, None, None, None))[1]
    )
    successful["context_length"] = successful["model"].apply(
        lambda m: PRICING_DATA.get(m, (None, None, None, None))[2]
    )
    successful["tier"] = successful["model"].apply(
        lambda m: PRICING_DATA.get(m, (None, None, None, None))[3]
    )
    
    # Calculate total cost (assuming 1:1 input/output ratio)
    successful["total_cost_per_1M"] = successful.apply(
        lambda row: get_total_cost_per_1M(row["input_cost"], row["output_cost"])
        if pd.notna(row["input_cost"]) and pd.notna(row["output_cost"])
        else None,
        axis=1
    )
    
    # Group by family
    families = successful.groupby("family")
    
    # Sort families alphabetically
    sorted_families = sorted(families.groups.keys())
    
    # Create organized output
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("WORKING MODELS ORGANIZED BY FAMILY AND COST")
    output_lines.append("=" * 100)
    output_lines.append("")
    
    for family in sorted_families:
        family_df = families.get_group(family)
        
        # Sort by total cost (ascending), then by model name
        family_df_sorted = family_df.sort_values(
            by=["total_cost_per_1M", "model"],
            ascending=[True, True],
            na_position="last"
        )
        
        output_lines.append(f"\n{'=' * 100}")
        output_lines.append(f"FAMILY: {family.upper()} ({len(family_df_sorted)} models)")
        output_lines.append(f"{'=' * 100}")
        output_lines.append("")
        
        # Group by cost tier for better readability
        free_models = family_df_sorted[family_df_sorted["total_cost_per_1M"] == 0.0]
        paid_models = family_df_sorted[family_df_sorted["total_cost_per_1M"] > 0.0]
        unknown_cost = family_df_sorted[family_df_sorted["total_cost_per_1M"].isna()]
        
        if len(free_models) > 0:
            output_lines.append("  FREE MODELS:")
            for _, row in free_models.iterrows():
                context = f"{int(row['context_length']/1000)}K" if pd.notna(row['context_length']) else "?"
                tier = row['tier'] if pd.notna(row['tier']) else "?"
                output_lines.append(f"    - {row['model']:<60} [Context: {context:>6}, Tier: {tier}]")
            output_lines.append("")
        
        if len(paid_models) > 0:
            output_lines.append("  PAID MODELS (sorted by cost):")
            for _, row in paid_models.iterrows():
                cost = f"${row['total_cost_per_1M']:.2f}"
                context = f"{int(row['context_length']/1000)}K" if pd.notna(row['context_length']) else "?"
                tier = row['tier'] if pd.notna(row['tier']) else "?"
                output_lines.append(
                    f"    - {row['model']:<60} [Cost: {cost:>8}/1M, Context: {context:>6}, Tier: {tier}]"
                )
            output_lines.append("")
        
        if len(unknown_cost) > 0:
            output_lines.append("  UNKNOWN COST:")
            for _, row in unknown_cost.iterrows():
                context = f"{int(row['context_length']/1000)}K" if pd.notna(row['context_length']) else "?"
                tier = row['tier'] if pd.notna(row['tier']) else "?"
                output_lines.append(f"    - {row['model']:<60} [Context: {context:>6}, Tier: {tier}]")
            output_lines.append("")
    
    # Print summary statistics
    output_lines.append("\n" + "=" * 100)
    output_lines.append("SUMMARY STATISTICS")
    output_lines.append("=" * 100)
    output_lines.append("")
    
    free_count = len(successful[successful["total_cost_per_1M"] == 0.0])
    paid_count = len(successful[successful["total_cost_per_1M"] > 0.0])
    unknown_count = len(successful[successful["total_cost_per_1M"].isna()])
    
    output_lines.append(f"Free models: {free_count}")
    output_lines.append(f"Paid models: {paid_count}")
    output_lines.append(f"Unknown cost: {unknown_count}")
    output_lines.append("")
    
    # Cost ranges
    if paid_count > 0:
        paid_df = successful[successful["total_cost_per_1M"] > 0.0]
        min_cost = paid_df["total_cost_per_1M"].min()
        max_cost = paid_df["total_cost_per_1M"].max()
        median_cost = paid_df["total_cost_per_1M"].median()
        output_lines.append(f"Cost range: ${min_cost:.2f} - ${max_cost:.2f} per 1M tokens")
        output_lines.append(f"Median cost: ${median_cost:.2f} per 1M tokens")
        output_lines.append("")
    
    # Print to console
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(output_text)
        print(f"\n✅ Results saved to {output_file}")
    
    # Also save as CSV organized by family and cost
    csv_output = successful.sort_values(
        by=["family", "total_cost_per_1M", "model"],
        ascending=[True, True, True],
        na_position="last"
    )
    
    csv_output_file = output_file.replace(".txt", "_organized.csv") if output_file else "smoke_test_organized.csv"
    csv_output.to_csv(csv_output_file, index=False)
    print(f"✅ Organized CSV saved to {csv_output_file}")
    
    return successful


if __name__ == "__main__":
    import sys
    
    csv_file = "chess_puzzles/data/smoke_test_results.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    analyze_smoke_test_results(csv_file, output_file)

