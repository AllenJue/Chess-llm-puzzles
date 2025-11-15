# Chess Puzzle Evaluator

A Python project for evaluating chess puzzles using multiple LLM paradigms (single model, self-consistency, and debate) across various models including OpenAI GPT-3.5-turbo-instruct and open-source models via Anannas API. The system evaluates models on Lichess puzzles and tracks accuracy, token usage, and error rates across different evaluation paradigms.

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY=<your key>
   ```
   Or create a `.env` file:
   ```env
   OPENAI_API_KEY=<your key>
   ```

## Usage

### Basic Commands

```bash
# Show help
python main.py --help

# Show puzzle statistics
python main.py --stats

# Evaluate 10 puzzles with GPT-3.5
python main.py --evaluate --max-puzzles 10

# Evaluate with GPT-4
python main.py --evaluate --max-puzzles 5 --model gpt-4-turbo

# Sample 100 puzzles and evaluate
python main.py --sample 100 --evaluate --max-puzzles 50

# Calculate Glicko-2 rating (after evaluation)
python main.py --rating
```

### Programmatic Usage

```python
from csv_reader import read_chess_puzzles_csv, sample_puzzles
from model_interface import ChessModelInterface
from chess_utils import build_chess_prompts
from glicko_rating import update_agent_rating_from_puzzles

# Load data
df = read_chess_puzzles_csv("lichess_puzzles_with_pgn_1000.csv")
df_sample = sample_puzzles(df, n=100)

# Initialize model
model = ChessModelInterface(api_key="your-api-key")

# Evaluate puzzles
# ... (see main.py for complete example)
```

## Results and Analysis

### Paradigm Comparison (First 50 Puzzles)

The system evaluates three paradigms:
- **Single Model**: Direct query to a single model
- **Self-Consistency**: Three independent queries (Aggressive, Positional, Neutral GMs) with consensus voting
- **Debate**: Two-agent debate (Affirmative vs Negative) with moderator/judge, with early consensus when both agree

#### Puzzle Accuracy Comparison
![Paradigm Comparison - Puzzle Accuracy](data/graphs/paradigm_comparison_puzzle_accuracy.png)

#### Move Accuracy Comparison
![Paradigm Comparison - Move Accuracy](data/graphs/paradigm_comparison_move_accuracy.png)

#### Token Usage Comparison
![Paradigm Comparison - Token Usage](data/graphs/paradigm_comparison_tokens.png)

### Sorted Performance Analysis

#### Puzzle Accuracy Sorted by Performance
All model-paradigm combinations sorted by puzzle accuracy (highest at top):
![Sorted Puzzle Accuracy](data/graphs/sorted_puzzle_accuracy.png)

#### Token Usage Sorted by Efficiency
All model-paradigm combinations sorted by tokens per move (highest at top):
![Sorted Token Usage](data/graphs/sorted_tokens_per_move.png)

### Key Findings

1. **Debate Efficiency**: The debate system achieves early consensus 81.8% of the time for GPT-3.5-turbo-instruct, using only 2 queries instead of 3, resulting in 0.58x the tokens of self-consistency.

2. **Self-Consistency Token Usage**: Self-consistency uses approximately 3x the prompt tokens of single model (as expected with 3 queries), but completion tokens vary based on response length.

3. **Model Performance**: Different models show varying accuracy across paradigms, with some models performing better in single model mode while others benefit from multi-agent approaches.

4. **Token Tracking**: The system now properly tracks tokens for all paradigms, even when queries fail to extract moves, ensuring accurate cost analysis.

## Data Source and Citations

This project uses chess puzzles sampled from the [Lichess Open Database](https://database.lichess.org/#puzzles), which contains over 5.4 million chess puzzles rated and tagged by the Lichess community.

The evaluation methodology follows established approaches for LLM chess evaluation, using similar prompting strategies as described in:
- [Chess LLM Evaluation](https://nicholas.carlini.com/writing/2023/chess-llm.html) by Nicholas Carlini
- [More Chess Analysis](https://dynomight.net/more-chess/) by Dynomight

**Current Models Evaluated**:
- GPT-3.5-turbo-instruct (OpenAI)
- arcee-ai/afm-4.5b
- deepseek-ai/deepseek-v3
- meta-llama/llama-3.1-8b-instruct
- meta-llama/llama-3.3-70b-instruct
- mistralai/mistral-small-24b-instruct-2501
- qwen/qwen3-235b-a22b-instruct-2507

**Evaluation Paradigms**: Single Model, Self-Consistency, Debate V2

**Results Location**: `data/test_results/` (organized by paradigm: `single_50/`, `self_consistency_50/`, `debate_50/`)

## Documentation

For detailed documentation, module descriptions, and technical details, see [DOCUMENTATION.md](DOCUMENTATION.md).