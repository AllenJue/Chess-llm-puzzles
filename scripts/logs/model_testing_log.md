# Model Testing Log

## Puzzle 0 Testing Results (2024-11-14)

### Summary
Tested 26 models (1 free + 25 low-cost open-source models) on puzzle 0 to identify which models can solve even an easy puzzle.

### Results

**✅ Models that solved puzzle 0:**
- `mistralai/mistral-small-24b-instruct-2501` - 1/1 correct moves

**❌ Models that failed puzzle 0 (25 models):**
- `together/ServiceNow-AI-Apriel-1.5-15b-Thinker` (FREE)
- `arcee-ai/afm-4.5b`
- `together/google-gemma-3n-E4B-it`
- `google/gemma-2-2b-it`
- `mistralai/mistral-7b-instruct-v0.3`
- `google/gemma-2-9b-it-fast`
- `qwen/qwen2.5-coder-7b`
- `meta-llama/llama-3.1-8b-instruct-fast`
- `together/meta-llama-Llama-3.2-3B-Instruct-Turbo`
- `openai/gpt-oss-20b`
- `together/meta-llama-Meta-Llama-3-8B-Instruct-Lite`
- `together/arize-ai-qwen-2-1.5b-instruct`
- `together/nvidia-NVIDIA-Nemotron-Nano-9B-v2`
- `openai/gpt-oss-120b`
- `together/arcee_ai-arcee-spotlight`
- `together/marin-community-marin-8b-instruct`
- `meta-llama/llama-3.1-8b-instruct`
- `together/meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo`
- `qwen/qwen3-30b-a3b-instruct-2507`
- `qwen/qwen3-32b`
- `mistralai/mistral-7b-instruct-v0.2`
- `together/togethercomputer-Refuel-Llm-V2-Small`
- `qwen/qwen2.5-vl-72b-instruct`
- `together/Qwen-Qwen3-235B-A22B-Instruct-2507-tput`
- `qwen/qwen3-235b-a22b-instruct-2507`

### Key Findings

1. **Only 1 out of 26 models solved puzzle 0**: `mistralai/mistral-small-24b-instruct-2501` ($0.13/1M tokens)

2. **Free models performance**: All free models tested previously failed to solve the first 50 puzzles. The free model tested here (`together/ServiceNow-AI-Apriel-1.5-15b-Thinker`) also failed puzzle 0.

3. **Open-source model limitations**: The vast majority of open-source models (including larger models like Qwen 235B) were unable to solve even an easy puzzle (puzzle 0), suggesting significant challenges with chess reasoning tasks.

4. **Cost vs. Performance**: The successful model (`mistralai/mistral-small-24b-instruct-2501`) is relatively low-cost at $0.13/1M tokens, making it a good candidate for further testing.

### Next Steps

- Test `mistralai/mistral-small-24b-instruct-2501` on the first 50 puzzles to evaluate its performance across a broader set of puzzles.

