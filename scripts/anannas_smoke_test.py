import argparse
import os
import re
import time
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


# All models from Anannas to test
ALL_MODELS: List[str] = [
    # Free models
    "google/gemini-2.0-flash-exp:free",
    "together/deepseek-ai-DeepSeek-R1-Distill-Llama-70B-free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "minimax/minimax-m2:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-2-9b-it:free",
    "google/gemma-3-4b-it:free",
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo-Free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "google/gemma-3-27b-it:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "qwen/qwen3-8b:free",
    "liquid/lfm-40b:free",
    
    # Paid models via Anannas
    "openai/gpt-oss-20b",
    "perplexity/llama-3.1-sonar-large-128k-online",
    "qwen/qwen3-next-80b-a3b-thinking",
    "perplexity/llama-3.1-sonar-small-128k-online",
    "allenai/molmo-7b-d",
    "jondurbin/airoboros-l2-70b",
    "anthropic/claude-3.7-sonnet",
    "embedding/multilingual-e5-large-instruct",
    "perplexity/r1-1776",
    "perplexity/sonar",
    "anthropic/claude-haiku-4.5",
    "alfredpros/codellama-7b-instruct-solidity",
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo-Free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "arcee-ai/maestro-reasoning",
    "amazon/nova-pro-v1",
    "openai/gpt-5-codex",
    "qwen/qwen3-next-80b-a3b-instruct",
    "all-hands/openhands-lm-32b-v0.1",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "meta-llama/llama-3.2-3b-instruct",
    "together/Qwen-Qwen3-235B-A22B-Instruct-2507-tput",
    "z-ai/glm-4.5-air",
    "openai/gpt-4o-mini-search-preview",
    "bytedance/seed-oss-36b-instruct",
    "allenai/llama-3.1-tulu-3-405b",
    "deepseek/deepseek-r1-0528",
    "mistralai/pixtral-large-2411",
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo",
    "anthropic/claude-3.5-sonnet",
    "mistralai/mistral-large-2411",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "qwen/qwen2.5-72b-instruct",
    "together/meta-llama-Meta-Llama-3.1-8B-Instruct-Reference",
    "openai/gpt-4.1",
    "perplexity/sonar-deep-research",
    "openai/gpt-5-chat",
    "meta-llama/llama-3.3-8b-instruct:free",
    "amazon/nova-lite-v1",
    "qwen/qwen3-coder",
    "allenai/olmo-7b-instruct",
    "google/gemma-2-9b-it-fast",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "meta-llama/llama-3.1-405b-instruct",
    "mistralai/mistral-7b-instruct-v0.2",
    "qwen/qwen2.5-coder-7b",
    "qwen/qwq-32b-preview",
    "arcee-ai/spotlight",
    "anthropic/claude-sonnet-4",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "google/gemini-2.0-flash-001",
    "google/gemma-3-12b-it",
    "alibaba/tongyi-deepresearch-30b-a3b",
    "google/gemma-2-9b-it",
    "anthropic/claude-opus-4",
    "meta-llama/llama-3.1-8b-instruct",
    "together/arize-ai-qwen-2-1.5b-instruct",
    "deepseek-ai/deepseek-v3",
    "nousresearch/deephermes-3-llama-3-8b-preview",
    "together/ServiceNow-AI-Apriel-1.5-15b-Thinker",
    "microsoft/mai-ds-r1",
    "google/gemini-2.5-flash-image-preview",
    "embedding/bge-large-en-v1.5",
    "cohere/command-r7b-12-2024",
    "google/gemini-flash-1.5-8b",
    "ai21/jamba-instruct",
    "mistralai/ministral-8b",
    "openai/gpt-4-turbo",
    "google/gemini-2.0-flash-lite-001",
    "anthracite-org/magnum-v2-72b",
    "google/gemini-2.5-pro-preview",
    "together/meta-llama-Llama-3.2-3B-Instruct-Turbo",
    "koboldai/psyfighter-13b-2",
    "anthropic/claude-3-sonnet",
    "meta-llama/llama-3.2-90b-vision-instruct",
    "together/Qwen-Qwen3-Coder-480B-A35B-Instruct-FP8",
    "mistralai/magistral-medium-2506",
    "z-ai/glm-4.5",
    "google/gemma-2-27b-it",
    "together/togethercomputer-MoA-1-Turbo",
    "together/deepseek-ai-DeepSeek-R1-0528-tput",
    "cohere/command-r-08-2024",
    "mistralai/mistral-tiny",
    "meituan/longcat-flash-chat",
    "nvidia/llama-3_1-nemotron-ultra-253b-v1",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-5.1-codex",
    "aion-labs/aion-rp-llama-3.1-8b",
    "mistralai/pixtral-12b",
    "embedding/nebius-qwen3-embedding-8b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "google/gemma-3-27b-it:free",
    "qwen/qwen-2.5-72b-instruct",
    "eva/eva-qwen-2.5-32b",
    "mattshumer/reflection-70b",
    "embedding/nebius-bge-multilingual-gemma2",
    "arcee-ai/coder-large",
    "samba-nova/llama-3.2-90b-vision-instruct",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "sao10k/l3.1-70b-hanami-x1",
    "qwen/qwq-32b",
    "neversleep/llama-3.1-lumimaid-70b",
    "together/meta-llama-Meta-Llama-3-8B-Instruct",
    "mistralai/ministral-3b",
    "huggingfaceh4/zephyr-7b-beta:free",
    "microsoft/phi-4-reasoning",
    "mistralai/mistral-medium-3.1",
    "openai/o1",
    "together/scb10x-scb10x-typhoon-2-1-gemma3-12b",
    "microsoft/phi-4",
    "perplexity/sonar-pro",
    "samba-nova/llama-3.2-11b-vision-instruct",
    "openai/gpt-4o-audio-preview",
    "nvidia/nemotron-nano-9b-v2",
    "moonshotai/kimi-k2-0711",
    "tencent/hunyuan-a13b-instruct",
    "openai/codex-mini",
    "nousresearch/hermes-4-70b",
    "qwen/qwen3-8b:free",
    "together/togethercomputer-MoA-1",
    "ai21/jamba-large-1.7",
    "google/gemini-2.5-pro",
    "qwen/qwen-2.5-vl-7b-instruct",
    "meta-llama/llama-4-maverick",
    "together/zai-org-GLM-4.5-Air-FP8",
    "cohere/command-a",
    "cohere/command-r",
    "meta-llama/llama-guard-4-12b",
    "openai/o4-mini",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-coder-plus",
    "inflection/inflection-3-pi",
    "openai/o3-mini",
    "eva-unit-01/eva-qwen-2.5-14b",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "google/gemma-3-4b-it",
    "grok/grok-2-vision-1212",
    "neversleep/llama-3.1-lumimaid-8b",
    "allenai/olmo-2-0325-32b-instruct",
    "meta-llama/llama-3.3-70b-instruct-fast",
    "mistralai/mistral-nemo",
    "together/Qwen-Qwen2.5-72B-Instruct-Turbo",
    "liquid/lfm-40b:free",
    "mistralai/mistral-medium-3",
    "sao10k/l3-euryale-70b",
    "qwen/qwen2.5-vl-32b-instruct",
    "openai/gpt-4o-2024-08-06",
    "meta-llama/llama-guard-3-8b",
    "openai/gpt-5.1",
    "together/meta-llama-Llama-3-70b-chat-hf",
    "anthropic/claude-3-haiku",
    "mistralai/mistral-small",
    "meta-llama/llama-3-70b-instruct",
    "01-ai/yi-large",
    "together/arcee_ai-arcee-spotlight",
    "qwen/qwen-plus",
    "nousresearch/hermes-2-theta-llama-3-8b",
    "cohere/command-r-plus-08-2024",
    "openai/gpt-4o",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "together/Qwen-Qwen2.5-7B-Instruct-Turbo",
    "together/mistralai-Mixtral-8x7B-Instruct-v0.1",
    "openai/gpt-5",
    "qwen/qwen-turbo",
    "deepseek/deepseek-v3.2-exp",
    "google/gemma-3-27b-it",
    "together/meta-llama-Meta-Llama-3.1-405B-Instruct-Lite-Pro",
    "grok/grok-code-fast-1",
    "together/google-gemma-3n-E4B-it",
    "qwen/qwen3-vl-8b-thinking",
    "arcee-ai/afm-4.5b",
    "amazon/nova-micro-v1",
    "together/meta-llama-Meta-Llama-3.1-405B-Instruct-Turbo",
    "01-ai/yi-large-fc",
    "inclusionai/ring-1t",
    "meta-llama/llama-3-8b-instruct",
    "anthropic/claude-3-opus",
    "openai/gpt-5.1-chat",
    "qwen/qwen2.5-vl-72b-instruct",
    "mistralai/devstral-small",
    "mistralai/mistral-7b-instruct-v0.1",
    "agentica-org/deepcoder-14b-preview",
    "together/deepcogito-cogito-v2-preview-deepseek-671b",
    "nousresearch/deephermes-3-mistral-24b-preview",
    "together/meta-llama-Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistralai/mistral-large",
    "openai/o3",
    "01-ai/yi-vision",
    "eva/eva-qwen-2.5-14b",
    "google/gemini-pro-1.5",
    "alpindale/goliath-120b",
    "openai/gpt-4o-2024-05-13",
    "google/gemini-2.0-flash-exp:free",
    "openai/gpt-4.1-nano",
    "qwen/qwen3-coder-flash",
    "perplexity/sonar-reasoning-pro",
    "together/deepseek-ai-DeepSeek-R1-Distill-Llama-70B-free",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "together/meta-llama-Llama-4-Scout-17B-16E-Instruct",
    "grok/grok-3-mini",
    "together/nvidia-NVIDIA-Nemotron-Nano-9B-v2",
    "embedding/gte-modernbert-base",
    "mistralai/mistral-large-2407",
    "google/gemma-2-2b-it",
    "qwen/qwen-2.5-7b-instruct",
    "microsoft/phi-3-mini-128k-instruct:free",
    "moonshotai/kimi-k2-instruct",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition",
    "deepseek-ai/deepseek-v3-0324",
    "openai/gpt-5.1-codex-mini",
    "google/gemini-2.5-flash",
    "together/Qwen-Qwen2.5-14B-Instruct",
    "qwen/qwen3-4b:free",
    "meta-llama/llama-3.1-405b",
    "meta-llama/llama-3.1-8b-instruct-fast",
    "embedding/m2-bert-80M-32k",
    "mistralai/magistral-small-2506",
    "nousresearch/hermes-3-llama-3.1-405b",
    "together/deepcogito-cogito-v2-preview-llama-109B-MoE",
    "microsoft/wizardlm-2-8x22b",
    "together/togethercomputer-Refuel-Llm-V2",
    "grok/grok-3",
    "qwen/qwen-2.5-coder-32b-instruct",
    "mistralai/codestral-2501",
    "together/meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistralai/mistral-7b-instruct-v0.3",
    "deepseek/deepseek-chat-v3.1",
    "together/deepseek-ai-DeepSeek-V3.1",
    "liquid/lfm-40b",
    "cognitivecomputations/dolphin-mixtral-8x7b",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "deepseek-ai/deepseek-v3-0324-fast",
    "qwen/qwen3-14b",
    "deepseek/deepseek-prover-v2",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "openai/gpt-4o-search-preview",
    "mistralai/mixtral-8x22b-instruct",
    "openai/gpt-4.1-mini",
    "anthropic/claude-sonnet-4.5",
    "together/meta-llama-Meta-Llama-3-70B-Instruct-Turbo",
    "embedding/nebius-bge-en-icl",
    "openai/gpt-4o:extended",
    "openai/chatgpt-4o-latest",
    "thedrummer/cydonia-24b-v4.1",
    "qwen/qwen-max",
    "microsoft/phi-3-medium-128k-instruct",
    "together/Qwen-Qwen3-235B-A22B-fp8-tput",
    "pygmalionai/pygmalion-13b-4bit",
    "together/moonshotai-Kimi-K2-Instruct-0905",
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo-batch",
    "anthropic/claude-opus-4.1",
    "cohere/command-r-plus",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "thedrummer/anubis-70b-v1.1",
    "together/togethercomputer-Refuel-Llm-V2-Small",
    "mistralai/mistral-small-3.1-24b-instruct",
    "aetherwiing/mn-starcannon-12b",
    "openai/o1-pro",
    "openai/gpt-4o-mini",
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "mistralai/mistral-7b-instruct",
    "openai/o1-mini-2024-09-12",
    "inflection/inflection-3-productivity",
    "qwen/qwen3-235b-a22b",
    "together/marin-community-marin-8b-instruct",
    "perplexity/llama-3.1-sonar-huge-128k-online",
    "grok/grok-4-fast-non-reasoning",
    "grok/grok-4",
    "mistralai/mixtral-8x7b-instruct",
    "qwen/qwen-vl-plus",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "moonshotai/kimi-k2-0905",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "minimax/minimax-m2:free",
    "openai/gpt-4o-mini-2024-07-18",
    "mistralai/mistral-saba",
    "stepfun-ai/step3",
    "qwen/qwen2.5-coder-7b-instruct",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.2-11b-vision-instruct",
    "google/gemma-3-12b-it:free",
    "moonshotai/kimi-k2-thinking",
    "nousresearch/hermes-4-405b",
    "grok/grok-4-fast-reasoning",
    "google/gemini-2.5-pro-preview-05-06",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "google/gemini-2.5-flash-lite",
    "perplexity/sonar-reasoning",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "google/gemma-2-9b-it:free",
    "liquid/lfm-7b",
    "openai/o1-mini",
    "z-ai/glm-4.6",
    "mistralai/devstral-medium",
    "deepseek/deepseek-r1",
    "openai/gpt-5-pro",
    "together/deepcogito-cogito-v2-preview-llama-405B",
    "nousresearch/hermes-2-pro-llama-3-8b",
    "qwen/qwen-vl-max",
    "grok/grok-4-0709",
    "arcee-ai/virtuoso-large",
    "samba-nova/meta-llama-3.2-3b-instruct",
    "qwen/qwen3-30b-a3b-thinking-2507",
    "baidu/ernie-4.5-300b-a47b",
    "samba-nova/meta-llama-3.2-1b-instruct",
    "qwen/qwen-plus-2025-07-28",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-chat-v3-0324",
    "embedding/bge-base-en-v1.5",
    "microsoft/phi-4-reasoning-plus",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-r1-distill-llama-70b",
    "qwen/qwen2.5-coder-7b-fast",
    "meta-llama/llama-3.1-70b-instruct",
    "embedding/nebius-e5-mistral-7b-instruct",
    "01-ai/yi-lightning",
    "deepseek/deepseek-v3.1-terminus",
    "minimax/minimax-m1",
    "nousresearch/hermes-3-llama-3.1-70b",
    "qwen/qwen3-235b-a22b-2507",
    "mistralai/mistral-small-24b-instruct-2501",
    "01-ai/yi-large-turbo",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "microsoft/phi-4-multimodal-instruct",
    "aion-labs/aion-1.0",
    "together/deepcogito-cogito-v2-preview-llama-70B",
    "microsoft/phi-3.5-mini-128k-instruct",
    "meta-llama/llama-guard-2-8b",
    "pygmalionai/mythalion-13b",
    "mistralai/devstral-small-2505",
    "mistralai/codestral-2508",
    "together/meta-llama-Llama-4-Maverick-17B-128E-Instruct-FP8",
    "inclusionai/ling-1t",
    "meta-llama/llama-4-scout",
    "openai/gpt-5-nano",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta-llama/llama-3.2-1b-instruct",
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-5-mini",
    "together/meta-llama-Meta-Llama-3.1-70B-Instruct-Reference",
    "deepseek/deepseek-v3",
    "openai/o3-pro",
    "qwen/qwen3-max",
    "together/meta-llama-Meta-Llama-3-8B-Instruct-Lite",
    "microsoft/phi-3-mini-128k-instruct",
    "alpindale/magnum-72b",
    "google/gemma-3-4b-it:free",
    "deepseek/deepseek-v3-0324",
    "openai/gpt-oss-20b",
    "perplexity/llama-3.1-sonar-large-128k-online",
    "qwen/qwen3-next-80b-a3b-thinking",
    "perplexity/llama-3.1-sonar-small-128k-online",
    "allenai/molmo-7b-d",
    "jondurbin/airoboros-l2-70b",
    "anthropic/claude-3.7-sonnet",
    "embedding/multilingual-e5-large-instruct",
    "perplexity/r1-1776",
    "perplexity/sonar",
    "anthropic/claude-haiku-4.5",
    "alfredpros/codellama-7b-instruct-solidity",
    "arcee-ai/maestro-reasoning",
    "amazon/nova-pro-v1",
    "qwen/qwen3-next-80b-a3b-instruct",
    "all-hands/openhands-lm-32b-v0.1",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "together/Qwen-Qwen3-235B-A22B-Instruct-2507-tput",
    "z-ai/glm-4.5-air",
    "openai/gpt-4o-mini-search-preview",
    "bytedance/seed-oss-36b-instruct",
    "allenai/llama-3.1-tulu-3-405b",
    "deepseek/deepseek-r1-0528",
    "mistralai/pixtral-large-2411",
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo",
    "mistralai/mistral-large-2411",
    "qwen/qwen2.5-72b-instruct",
    "together/meta-llama-Meta-Llama-3.1-8B-Instruct-Reference",
    "openai/gpt-4.1",
    "perplexity/sonar-deep-research",
    "openai/gpt-5-chat",
    "amazon/nova-lite-v1",
    "qwen/qwen3-coder",
    "allenai/olmo-7b-instruct",
    "google/gemma-2-9b-it-fast",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "meta-llama/llama-3.1-405b-instruct",
    "mistralai/mistral-7b-instruct-v0.2",
    "qwen/qwen2.5-coder-7b",
    "qwen/qwq-32b-preview",
    "arcee-ai/spotlight",
    "anthropic/claude-sonnet-4",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "google/gemini-2.0-flash-001",
    "google/gemma-3-12b-it",
    "alibaba/tongyi-deepresearch-30b-a3b",
    "google/gemma-2-9b-it",
    "anthropic/claude-opus-4",
    "meta-llama/llama-3.1-8b-instruct",
    "together/arize-ai-qwen-2-1.5b-instruct",
    "deepseek-ai/deepseek-v3",
    "nousresearch/deephermes-3-llama-3-8b-preview",
    "together/ServiceNow-AI-Apriel-1.5-15b-Thinker",
    "microsoft/mai-ds-r1",
    "google/gemini-2.5-flash-image-preview",
    "embedding/bge-large-en-v1.5",
    "cohere/command-r7b-12-2024",
    "google/gemini-flash-1.5-8b",
    "ai21/jamba-instruct",
    "mistralai/ministral-8b",
    "google/gemini-2.0-flash-lite-001",
    "anthracite-org/magnum-v2-72b",
    "google/gemini-2.5-pro-preview",
    "together/meta-llama-Llama-3.2-3B-Instruct-Turbo",
    "koboldai/psyfighter-13b-2",
    "together/Qwen-Qwen3-Coder-480B-A35B-Instruct-FP8",
    "mistralai/magistral-medium-2506",
    "z-ai/glm-4.5",
    "google/gemma-2-27b-it",
    "together/togethercomputer-MoA-1-Turbo",
    "together/deepseek-ai-DeepSeek-R1-0528-tput",
    "cohere/command-r-08-2024",
    "mistralai/mistral-tiny",
    "meituan/longcat-flash-chat",
    "nvidia/llama-3_1-nemotron-ultra-253b-v1",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-5.1-codex",
    "aion-labs/aion-rp-llama-3.1-8b",
    "mistralai/pixtral-12b",
    "embedding/nebius-qwen3-embedding-8b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "eva/eva-qwen-2.5-32b",
    "mattshumer/reflection-70b",
    "embedding/nebius-bge-multilingual-gemma2",
    "arcee-ai/coder-large",
    "samba-nova/llama-3.2-90b-vision-instruct",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "sao10k/l3.1-70b-hanami-x1",
    "qwen/qwq-32b",
    "neversleep/llama-3.1-lumimaid-70b",
    "together/meta-llama-Meta-Llama-3-8B-Instruct",
    "mistralai/ministral-3b",
    "microsoft/phi-4-reasoning",
    "mistralai/mistral-medium-3.1",
    "together/scb10x-scb10x-typhoon-2-1-gemma3-12b",
    "microsoft/phi-4",
    "perplexity/sonar-pro",
    "samba-nova/llama-3.2-11b-vision-instruct",
    "openai/gpt-4o-audio-preview",
    "nvidia/nemotron-nano-9b-v2",
    "moonshotai/kimi-k2-0711",
    "tencent/hunyuan-a13b-instruct",
    "openai/codex-mini",
    "nousresearch/hermes-4-70b",
    "together/togethercomputer-MoA-1",
    "ai21/jamba-large-1.7",
    "google/gemini-2.5-pro",
    "qwen/qwen-2.5-vl-7b-instruct",
    "meta-llama/llama-4-maverick",
    "together/zai-org-GLM-4.5-Air-FP8",
    "cohere/command-a",
    "cohere/command-r",
    "meta-llama/llama-guard-4-12b",
    "openai/o4-mini",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-coder-plus",
    "inflection/inflection-3-pi",
    "openai/o3-mini",
    "eva-unit-01/eva-qwen-2.5-14b",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "google/gemma-3-4b-it",
    "grok/grok-2-vision-1212",
    "neversleep/llama-3.1-lumimaid-8b",
    "allenai/olmo-2-0325-32b-instruct",
    "meta-llama/llama-3.3-70b-instruct-fast",
    "mistralai/mistral-nemo",
    "together/Qwen-Qwen2.5-72B-Instruct-Turbo",
    "mistralai/mistral-medium-3",
    "sao10k/l3-euryale-70b",
    "qwen/qwen2.5-vl-32b-instruct",
    "openai/gpt-4o-2024-08-06",
    "meta-llama/llama-guard-3-8b",
    "openai/gpt-5.1",
    "together/meta-llama-Llama-3-70b-chat-hf",
    "anthropic/claude-3-haiku",
    "mistralai/mistral-small",
    "meta-llama/llama-3-70b-instruct",
    "01-ai/yi-large",
    "together/arcee_ai-arcee-spotlight",
    "qwen/qwen-plus",
    "nousresearch/hermes-2-theta-llama-3-8b",
    "cohere/command-r-plus-08-2024",
    "openai/gpt-4o",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "together/Qwen-Qwen2.5-7B-Instruct-Turbo",
    "together/mistralai-Mixtral-8x7B-Instruct-v0.1",
    "openai/gpt-5",
    "qwen/qwen-turbo",
    "deepseek/deepseek-v3.2-exp",
    "together/meta-llama-Meta-Llama-3.1-405B-Instruct-Lite-Pro",
    "grok/grok-code-fast-1",
    "together/google-gemma-3n-E4B-it",
    "qwen/qwen3-vl-8b-thinking",
    "arcee-ai/afm-4.5b",
    "amazon/nova-micro-v1",
    "together/meta-llama-Meta-Llama-3.1-405B-Instruct-Turbo",
    "01-ai/yi-large-fc",
    "inclusionai/ring-1t",
    "meta-llama/llama-3-8b-instruct",
    "anthropic/claude-3-opus",
    "openai/gpt-5.1-chat",
    "qwen/qwen2.5-vl-72b-instruct",
    "mistralai/devstral-small",
    "mistralai/mistral-7b-instruct-v0.1",
    "agentica-org/deepcoder-14b-preview",
    "together/deepcogito-cogito-v2-preview-deepseek-671b",
    "nousresearch/deephermes-3-mistral-24b-preview",
    "together/meta-llama-Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistralai/mistral-large",
    "openai/o3",
    "01-ai/yi-vision",
    "eva/eva-qwen-2.5-14b",
    "google/gemini-pro-1.5",
    "alpindale/goliath-120b",
    "openai/gpt-4o-2024-05-13",
    "openai/gpt-4.1-nano",
    "qwen/qwen3-coder-flash",
    "perplexity/sonar-reasoning-pro",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "together/meta-llama-Llama-4-Scout-17B-16E-Instruct",
    "grok/grok-3-mini",
    "together/nvidia-NVIDIA-Nemotron-Nano-9B-v2",
    "embedding/gte-modernbert-base",
    "mistralai/mistral-large-2407",
    "google/gemma-2-2b-it",
    "qwen/qwen-2.5-7b-instruct",
    "moonshotai/kimi-k2-instruct",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition",
    "deepseek-ai/deepseek-v3-0324",
    "openai/gpt-5.1-codex-mini",
    "google/gemini-2.5-flash",
    "together/Qwen-Qwen2.5-14B-Instruct",
    "meta-llama/llama-3.1-405b",
    "meta-llama/llama-3.1-8b-instruct-fast",
    "embedding/m2-bert-80M-32k",
    "mistralai/magistral-small-2506",
    "nousresearch/hermes-3-llama-3.1-405b",
    "together/deepcogito-cogito-v2-preview-llama-109B-MoE",
    "microsoft/wizardlm-2-8x22b",
    "together/togethercomputer-Refuel-Llm-V2",
    "grok/grok-3",
    "mistralai/codestral-2501",
    "deepseek/deepseek-chat-v3.1",
    "together/deepseek-ai-DeepSeek-V3.1",
    "cognitivecomputations/dolphin-mixtral-8x7b",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "deepseek-ai/deepseek-v3-0324-fast",
    "qwen/qwen3-14b",
    "deepseek/deepseek-prover-v2",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "openai/gpt-4o-search-preview",
    "mistralai/mixtral-8x22b-instruct",
    "openai/gpt-4.1-mini",
    "together/meta-llama-Meta-Llama-3-70B-Instruct-Turbo",
    "embedding/nebius-bge-en-icl",
    "openai/gpt-4o:extended",
    "openai/chatgpt-4o-latest",
    "thedrummer/cydonia-24b-v4.1",
    "microsoft/phi-3-medium-128k-instruct",
    "together/Qwen-Qwen3-235B-A22B-fp8-tput",
    "pygmalionai/pygmalion-13b-4bit",
    "together/moonshotai-Kimi-K2-Instruct-0905",
    "together/meta-llama-Llama-3.3-70B-Instruct-Turbo-batch",
    "cohere/command-r-plus",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "thedrummer/anubis-70b-v1.1",
    "together/togethercomputer-Refuel-Llm-V2-Small",
    "mistralai/mistral-small-3.1-24b-instruct",
    "aetherwiing/mn-starcannon-12b",
    "qwen/qwen3-vl-8b-instruct",
    "openai/o1-mini-2024-09-12",
    "inflection/inflection-3-productivity",
    "qwen/qwen3-235b-a22b",
    "together/marin-community-marin-8b-instruct",
    "grok/grok-4-fast-non-reasoning",
    "grok/grok-4",
    "mistralai/mixtral-8x7b-instruct",
    "qwen/qwen-vl-plus",
    "moonshotai/kimi-k2-0905",
    "minimax/minimax-m2:free",
    "openai/gpt-4o-mini-2024-07-18",
    "mistralai/mistral-saba",
    "stepfun-ai/step3",
    "qwen/qwen2.5-coder-7b-instruct",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.2-11b-vision-instruct",
    "moonshotai/kimi-k2-thinking",
    "nousresearch/hermes-4-405b",
    "grok/grok-4-fast-reasoning",
    "google/gemini-2.5-pro-preview-05-06",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "google/gemini-2.5-flash-lite",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "liquid/lfm-7b",
    "z-ai/glm-4.6",
    "mistralai/devstral-medium",
    "together/deepcogito-cogito-v2-preview-llama-405B",
    "nousresearch/hermes-2-pro-llama-3-8b",
    "grok/grok-4-0709",
    "arcee-ai/virtuoso-large",
    "samba-nova/meta-llama-3.2-3b-instruct",
    "qwen/qwen3-30b-a3b-thinking-2507",
    "baidu/ernie-4.5-300b-a47b",
    "samba-nova/meta-llama-3.2-1b-instruct",
    "qwen/qwen-plus-2025-07-28",
    "embedding/bge-base-en-v1.5",
    "microsoft/phi-4-reasoning-plus",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-r1-distill-llama-70b",
    "qwen/qwen2.5-coder-7b-fast",
    "embedding/nebius-e5-mistral-7b-instruct",
    "01-ai/yi-lightning",
    "deepseek/deepseek-v3.1-terminus",
    "minimax/minimax-m1",
    "nousresearch/hermes-3-llama-3.1-70b",
    "qwen/qwen3-235b-a22b-2507",
    "mistralai/mistral-small-24b-instruct-2501",
    "01-ai/yi-large-turbo",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "microsoft/phi-4-multimodal-instruct",
    "aion-labs/aion-1.0",
    "together/deepcogito-cogito-v2-preview-llama-70B",
    "microsoft/phi-3.5-mini-128k-instruct",
    "meta-llama/llama-guard-2-8b",
    "pygmalionai/mythalion-13b",
    "mistralai/devstral-small-2505",
    "mistralai/codestral-2508",
    "together/meta-llama-Llama-4-Maverick-17B-128E-Instruct-FP8",
    "inclusionai/ling-1t",
    "meta-llama/llama-4-scout",
    "openai/gpt-5-nano",
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-5-mini",
    "together/meta-llama-Meta-Llama-3.1-70B-Instruct-Reference",
    "openai/o3-pro",
    "together/meta-llama-Meta-Llama-3-8B-Instruct-Lite",
    "alpindale/magnum-72b",
    "deepseek/deepseek-v3-0324",
]


def get_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def run_smoke_test(client: OpenAI, model: str, max_retries: int = 2) -> dict:
    """Run smoke test on a model. Returns dict with status."""
    print(f"\n=== Testing model: {model} ===")
    
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
                        return {"model": model, "status": "success", "response_type": "reasoning", "message": final_answer}
                    else:
                        # Show more of the reasoning to help debug
                        preview_len = 300 if finish_reason == "length" else 150
                        print(f"⚠️  Warning: Model returned empty content but has reasoning")
                        print(f"   Finish reason: {finish_reason}")
                        if finish_reason == "length":
                            print(f"   Note: Hit token limit - may need more tokens for reasoning models")
                        print(f"   Reasoning preview: {reasoning_text[:preview_len]}...")
                        print(f"✅ API call succeeded (model used reasoning mode, no clear final answer extracted)")
                        return {"model": model, "status": "success", "response_type": "reasoning_empty", "message": "Reasoning mode but no clear answer"}
                else:
                    print(f"⚠️  Warning: Model returned empty response")
                    print(f"   Finish reason: {finish_reason}")
                    print(f"✅ API call succeeded (but response is empty)")
                    return {"model": model, "status": "success", "response_type": "empty", "message": "Empty response"}
            else:
                print(f"✅ Success: {reply}")
                return {"model": model, "status": "success", "response_type": "content", "message": reply}
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
                return {"model": model, "status": "failed", "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test all models from Anannas API.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Models to test (defaults to all models from list).",
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
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file to save results (optional).",
    )

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("ANANNAS_API_KEY not provided. Set the env var or use --api-key.")

    # Determine which models to test
    if args.models:
        models = args.models
    else:
        models = ALL_MODELS
        print(f"Testing {len(models)} models from Anannas API")
    
    client = get_client(api_key=args.api_key, base_url=args.base_url)
    
    results = []
    successful = []
    failed = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}]")
        result = run_smoke_test(client, model)
        results.append(result)
        
        if result["status"] == "success":
            successful.append(model)
        else:
            failed.append(model)
        
        # Delay between models
        if i < len(models):
            time.sleep(args.delay)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total models tested: {len(models)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n✅ WORKING MODELS ({len(successful)}):")
        for model in successful:
            result = next(r for r in results if r["model"] == model)
            response_type = result.get("response_type", "unknown")
            print(f"   - {model} ({response_type})")
    
    if failed:
        print(f"\n❌ FAILED MODELS ({len(failed)}):")
        for model in failed:
            result = next(r for r in results if r["model"] == model)
            error = result.get("error", "Unknown error")
            # Truncate long error messages
            if len(error) > 100:
                error = error[:100] + "..."
            print(f"   - {model}: {error}")
    
    # Save to CSV if requested
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
