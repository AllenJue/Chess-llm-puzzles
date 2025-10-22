# Chess Puzzle Evaluator - Documentation

## Project Structure

```
chess_puzzles/
├── .gitignore                 # Git ignore file
├── env_example.txt           # Environment variables template
├── requirements.txt          # Python dependencies
├── README.md                # Quick start guide
├── DOCUMENTATION.md         # This file
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

This project uses chess puzzles sampled from the [Lichess Open Database](https://database.lichess.org/#puzzles), which contains over 5.4 million chess puzzles rated and tagged by the Lichess community. The puzzles are generated from real games and analyzed with Stockfish NNUE at 40 meganodes, providing high-quality tactical positions for evaluation.

The evaluation methodology follows established approaches for LLM chess evaluation, using similar prompting strategies as described in:
- [Chess LLM Evaluation](https://nicholas.carlini.com/writing/2023/chess-llm.html) by Nicholas Carlini
- [More Chess Analysis](https://dynomight.net/more-chess/) by Dynomight

**Current Model**: GPT-3.5-turbo-instruct  
**Future Plans**: Evaluation of additional models including GPT-4, Claude, and other chess-specialized models

## Related Work and Limitations

This project addresses limitations in previous chess puzzle evaluation approaches. Previous work, such as the [llm-chess-puzzles repository](https://github.com/kagisearch/llm-chess-puzzles/blob/main/llmchess.py), often does not test puzzles in their fullest capacity. Common limitations include:

- **Incomplete puzzle evaluation**: Many existing approaches only test single moves rather than complete puzzle sequences
- **Limited puzzle coverage**: Previous work often focuses on simple tactical patterns rather than complex multi-move combinations
- **Inadequate rating systems**: Most existing evaluations lack proper chess rating systems like Glicko-2 for meaningful performance assessment
- **Simplified board states**: Some approaches don't properly handle complex board positions or partial game states

This project addresses these limitations by:
- **Full puzzle sequences**: Evaluating complete puzzle solutions, not just single moves
- **Comprehensive testing**: Using real Lichess puzzle data with proper game contexts
- **Proper rating system**: Implementing Glicko-2 rating for accurate performance measurement
- **Real game positions**: Using actual game positions from Lichess with proper PGN context

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
