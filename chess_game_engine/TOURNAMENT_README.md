# Chess Tournament System

A comprehensive tournament system for evaluating LLMs, Stockfish, and humans in chess matches, similar to the Kaggle Chess benchmark.

## Features

- **Model vs Model Tournaments**: Round-robin tournaments between multiple LLMs
- **Stockfish Baseline**: Include Stockfish as a baseline player for comparison
- **Human vs LLM**: Play against any LLM model interactively
- **Opening Variations**: Test models from common opening positions (like Kaggle benchmark)
- **Glicko-2 Ratings**: Sophisticated rating system that handles uncertainty
- **Comprehensive Tracking**: Saves PGN files, JSON results, and rating history

## Quick Start

### 0. Minimal Test (Recommended First)

```bash
# Test with minimal setup (1 game per matchup, 2 players)
python test_minimal_tournament.py
```

### 1. Run a Full Tournament

```bash
# Run tournament with default models (GPT-3.5, DeepSeek-V3, Mistral-24B, Llama-3.3-70B)
python run_tournament.py --tournament

# Include Stockfish as baseline
python run_tournament.py --tournament --include-stockfish --stockfish-skill 10

# Use opening variations (like Kaggle benchmark)
python run_tournament.py --tournament --use-openings

# More games per matchup for better statistics
python run_tournament.py --tournament --games-per-matchup 4
```

### 2. Play Against an LLM

```bash
# Play as white against GPT-3.5
python run_tournament.py --human-vs-llm --model gpt-3.5-turbo-instruct

# Play as black against DeepSeek-V3
python run_tournament.py --human-vs-llm --model deepseek-ai/deepseek-v3 --human-black

# Start from a specific opening
python run_tournament.py --human-vs-llm --model gpt-3.5-turbo-instruct --use-openings --opening e4_e5
```

### 3. Custom Tournament

```bash
# Specify which models to include
python run_tournament.py --tournament \
    --models gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 \
    --model-names "GPT-3.5" "DeepSeek-V3" \
    --include-stockfish \
    --games-per-matchup 2
```

## Default Models

The system includes these models by default:

1. **GPT-3.5-turbo-instruct** (OpenAI)
2. **DeepSeek-V3** (`deepseek-ai/deepseek-v3`) - 67B parameters
3. **Mistral-Small-24B** (`mistralai/mistral-small-24b-instruct-2501`) - 24B parameters
4. **Llama-3.3-70B** (`meta-llama/llama-3.3-70b-instruct`) - 70B parameters

## Rating System: Glicko-2

We use **Glicko-2** instead of simple Elo because:

1. **Handles Uncertainty**: Tracks Rating Deviation (RD) - models with fewer games have higher uncertainty
2. **More Accurate**: Better for tournaments with varying numbers of games per player
3. **Already Implemented**: Your codebase already has Glicko-2
4. **Compatible**: Glicko-2 mu (rating) is approximately equivalent to Elo rating

### Rating Interpretation

- **Rating (mu)**: Similar to Elo rating (default start: 1500)
- **Rating Deviation (RD/phi)**: Uncertainty measure (lower = more confident)
- **Volatility (sigma)**: How much the rating fluctuates

## Opening Variations

The system includes 8 common two-ply opening positions:

- `e4_e5`: King's Pawn Game
- `e4_c5`: Sicilian Defense
- `e4_e6`: French Defense
- `d4_d5`: Queen's Gambit Declined
- `d4_Nf6`: Indian Defense
- `Nf3_d5`: Reti Opening
- `c4_e5`: English Opening
- `e4_c6`: Caro-Kann Defense

## Output Files

Tournament results are saved to `data/tournaments/`:

- `*.json`: Individual game results with full metadata
- `*.pgn`: Standard chess notation files (importable to chess software)
- `ratings.json`: Complete rating history and current ratings
- `tournament_summary.json`: Tournament overview and leaderboard

## Command Line Options

### Tournament Options

```
--tournament                    Run full tournament
--games-per-matchup N          Number of games per matchup (default: 2)
--use-openings                  Use opening variations
--opening KEY                   Use specific opening (e.g., 'e4_e5')
--openings KEY1 KEY2 ...        Use multiple specific openings
--include-stockfish             Include Stockfish baseline
--stockfish-skill N             Stockfish skill level 0-20 (default: 5)
--output-dir DIR                Output directory (default: data/tournaments)
--quiet                         Reduce verbosity
```

### Player Selection

```
--models MODEL1 MODEL2 ...      Specific models to include
--model-names NAME1 NAME2 ...   Display names for models
```

### Human vs LLM

```
--human-vs-llm                 Play against an LLM
--model MODEL                   Model to play against
--human-black                   Human plays black (default: white)
--model-name NAME               Display name for model
```

## Programmatic Usage

### Create a Custom Tournament

```python
from model_vs_model_game import Player, PlayerType
from tournament_manager import TournamentManager

# Create players
players = [
    Player(
        name="GPT-3.5",
        player_type=PlayerType.LLM,
        model_name="gpt-3.5-turbo-instruct",
    ),
    Player(
        name="DeepSeek-V3",
        player_type=PlayerType.LLM,
        model_name="deepseek-ai/deepseek-v3",
    ),
    Player(
        name="Stockfish",
        player_type=PlayerType.STOCKFISH,
        stockfish_skill=10,
    ),
]

# Create tournament manager
manager = TournamentManager(
    players=players,
    games_per_matchup=2,
    use_openings=True,
    output_dir="data/my_tournament",
)

# Run tournament
summary = manager.run_tournament()
```

### Play Human vs LLM Programmatically

```python
from model_vs_model_game import Player, PlayerType, ModelVsModelGame

def get_human_move(board):
    print(board)
    san_move = input("Your move (SAN): ")
    return san_move

human = Player(
    name="Human",
    player_type=PlayerType.HUMAN,
    human_move_callback=get_human_move,
)

llm = Player(
    name="GPT-3.5",
    player_type=PlayerType.LLM,
    model_name="gpt-3.5-turbo-instruct",
)

game = ModelVsModelGame(white_player=human, black_player=llm)
result = game.play_game()
```

## Stockfish Configuration

Stockfish skill levels:

- **0-3**: Beginner (good for testing)
- **4-7**: Intermediate (recommended for LLMs)
- **8-12**: Advanced
- **13-17**: Expert
- **18-20**: Maximum strength (grandmaster level)

Default: Skill level 5 (intermediate) - provides good challenge without being overwhelming.

## API Configuration

Set environment variables:

```bash
# For OpenAI models
export OPENAI_API_KEY="your-key"

# For Anannas API (open-source models)
export ANANNAS_API_KEY="your-key"
export ANANNAS_API_URL="https://api.anannas.ai/v1"
```

Or use command-line options:

```bash
python run_tournament.py --tournament --api-key YOUR_KEY --base-url YOUR_URL
```

## Example Tournament Results

After running a tournament, you'll see:

```
================================================================================
Tournament Leaderboard
================================================================================
Rank   Player                                    Rating       RD           Games
--------------------------------------------------------------------------------
1      Stockfish                                 1850.2       45.3           20
2      GPT-3.5-turbo-instruct                    1620.5       78.2           20
3      DeepSeek-V3                               1580.3       82.1           20
4      Llama-3.3-70B                             1520.1       95.4           20
5      Mistral-Small-24B                         1480.7      102.3           20
================================================================================
```

## Comparison with Kaggle Benchmark

This system is similar to the Kaggle Chess benchmark:

| Feature | Kaggle | This System |
|---------|--------|-------------|
| Model vs Model | ✅ | ✅ |
| Opening Variations | ✅ | ✅ |
| Rating System | Bradley-Terry/Elo | Glicko-2 |
| Stockfish Baseline | ✅ | ✅ |
| PGN Export | ✅ | ✅ |
| Human Play | ❌ | ✅ |

**Advantages of this system:**
- Human vs LLM capability
- Glicko-2 handles uncertainty better
- More flexible tournament formats
- Programmatic API

## Troubleshooting

### Stockfish Not Found
```bash
# Install Stockfish
# macOS:
brew install stockfish

# Ubuntu/Debian:
sudo apt-get install stockfish

# Or specify path:
python run_tournament.py --tournament --stockfish-path /path/to/stockfish
```

### API Errors
- Check your API key is set correctly
- Ensure you have sufficient API credits
- Check rate limits for your API provider

### Model Not Found
- Verify model name matches your API provider's naming
- For Anannas API, use format: `provider/model-name`
- Check API documentation for exact model identifiers

## Next Steps

1. Run a small test tournament: `python run_tournament.py --tournament --games-per-matchup 1`
2. Try human vs LLM: `python run_tournament.py --human-vs-llm --model gpt-3.5-turbo-instruct`
3. Analyze results in `data/tournaments/`
4. Compare ratings across different tournament runs

