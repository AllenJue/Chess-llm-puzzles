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

## Notes

- The project was converted from a Jupyter notebook
- Google Colab dependencies have been replaced with local file handling
- Environment variables replace Colab's userdata system
- All modules are properly documented and tested
- Glicko-2 implementation includes proper attribution

## Troubleshooting

- Ensure OpenAI API key is set correctly
- Check that CSV file exists and is readable
- Verify all dependencies are installed
- For API rate limits, adjust `API_DELAY` in environment
