# Chess Puzzle Evaluator

A Python project for evaluating chess puzzles using OpenAI models and the Glicko-2 rating system.

## Project Structure

```
chess_puzzles/
├── .gitignore                 # Git ignore file
├── env_example.txt           # Environment variables template
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── main.py                  # Main entry point
├── csv_reader.py            # CSV data handling
├── chess_utils.py           # Chess-related utilities
├── model_interface.py       # OpenAI API interface
├── glicko_rating.py         # Glicko-2 rating system
└── lichess_puzzles_with_pgn_1000.csv  # Sample data
```

## Features

- **CSV Data Management**: Read and process chess puzzle datasets
- **Chess Utilities**: PGN processing, move extraction, board state management
- **Model Interface**: OpenAI API integration for move prediction
- **Glicko-2 Rating**: Implement chess rating system with proper attribution
- **Evaluation Pipeline**: Complete puzzle evaluation workflow

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp env_example.txt .env
   # Edit .env with your OpenAI API key
   ```

## Usage

### Basic Usage

```bash
# Show help
python main.py --help

# Show puzzle statistics
python main.py --csv-file lichess_puzzles_with_pgn_1000.csv --stats

# Evaluate 10 puzzles with GPT-3.5
python main.py --evaluate --max-puzzles 10 --model gpt-3.5-turbo-instruct

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

## Modules

### csv_reader.py
- `read_chess_puzzles_csv()`: Load puzzle data from CSV
- `sample_puzzles()`: Random sampling of puzzles
- `get_puzzle_stats()`: Basic statistics
- `filter_puzzles_by_rating()`: Filter by rating range
- `filter_puzzles_by_theme()`: Filter by puzzle themes

### chess_utils.py
- `extract_game_id_and_move()`: Parse Lichess URLs
- `get_clean_pgn()`: Fetch PGN from Lichess API
- `build_chess_prompts()`: Create model prompts
- `extract_predicted_move()`: Extract SAN moves from model responses
- `san_to_uci()`: Convert SAN to UCI notation

### model_interface.py
- `ChessModelInterface`: Main class for OpenAI API interactions
- `query_model_for_move()`: Query GPT-3.5-turbo-instruct
- `query_model_for_gpt4_move()`: Query GPT-4-turbo
- Rate limiting and error handling

### glicko_rating.py
- `Rating`: Glicko-2 rating class
- `Glicko2`: Rating system implementation
- `update_agent_rating_from_puzzles()`: Update ratings from puzzle results
- Proper attribution to original author (Heungsub Lee)

## Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
CSV_FILE_PATH=lichess_puzzles_with_pgn_1000.csv
MODEL_NAME=gpt-3.5-turbo-instruct
MAX_PUZZLES=1000
API_DELAY=0.1
```

## Dependencies

- `pandas`: Data manipulation
- `python-chess`: Chess library
- `python-lichess`: Lichess API client
- `openai`: OpenAI API client
- `requests`: HTTP requests
- `python-dotenv`: Environment variables
- `matplotlib`, `seaborn`: Visualization (optional)

## Glicko-2 Attribution

The Glicko-2 rating system implementation is based on the work by Heungsub Lee:
- Original repository: https://github.com/heungsub/glicko2
- Copyright (c) 2012 by Heungsub Lee
- License: BSD

## Example Workflow

1. **Load Data**: Read chess puzzle CSV file
2. **Sample**: Select random subset of puzzles
3. **Evaluate**: Run puzzles through OpenAI model
4. **Rate**: Calculate Glicko-2 rating based on performance
5. **Analyze**: Generate statistics and visualizations

## Data Source and Methodology

This project uses chess puzzles sampled from the [Lichess Open Database](https://database.lichess.org/#puzzles), which contains over 5.4 million chess puzzles rated and tagged by the Lichess community. 

The evaluation methodology follows established approaches for LLM chess evaluation, using similar prompting strategies as described in:
- [Chess LLM Evaluation](https://nicholas.carlini.com/writing/2023/chess-llm.html) by Nicholas Carlini
- [More Chess Analysis](https://dynomight.net/more-chess/) by Dynomight

**Current Model**: GPT-3.5-turbo-instruct  
**Future Plans**: Evaluation of additional models including GPT-4, Claude, and other chess-specialized models

## Results and Analysis

### Performance by Puzzle Theme
![Accuracy by Puzzle Theme](graphs/accuracy_by_puzzle_theme.png)

### Performance by Puzzle Rating
![Accuracy by Puzzle Rating](graphs/accuracy_puzzle_bin.png)

## Related Work and Limitations

This project addresses limitations in previous chess puzzle evaluation approaches. Previous work, such as the [llm-chess-puzzles repository](https://github.com/kagisearch/llm-chess-puzzles/blob/main/llmchess.py), often does not test puzzles in their fullest capacity. Common limitations include:

- **Incomplete puzzle evaluation**: Many existing approaches only test single moves rather than complete puzzle sequences

