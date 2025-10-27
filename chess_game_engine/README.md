# Chess Game Engine

Play full chess games against Stockfish using LLM models with single model or self-consistency approaches.

## Features

- **Single Model**: Use a single LLM call to generate moves
- **Self-Consistency**: Use 3 independent LLM queries with majority voting
- **Stockfish Integration**: Play against the Stockfish chess engine
- **Fallback System**: Automatically uses random legal moves when AI fails
- **Game Recording**: Save games as JSON and PGN formats
- **Flexible Configuration**: Choose model, time limits, and colors
- **Move Tracking**: Records when fallback moves were used and why

## Requirements

- Python 3.7+
- `python-chess` library
- `stockfish` chess engine installed
- OpenAI API key

## Installation

```bash
# Install required packages
pip install python-chess openai

# Install Stockfish (varies by system)
# Ubuntu/Debian:
sudo apt-get install stockfish

# macOS:
brew install stockfish

# Windows: Download from https://stockfishchess.org/download/
```

## Usage

### Basic Usage

```bash
# Play with single model as white (Stockfish skill level 5/20)
python chess_game.py

# Play with self-consistency as black (easier Stockfish)
python chess_game.py --self-consistency --model-color black --skill 3

# Use GPT-4 with custom Stockfish time and skill level
python chess_game.py --model gpt-4-turbo --time 2.0 --skill 8

# Play against random legal moves instead of Stockfish
python chess_game.py --random-opponent --save-pgn

# Self-consistency vs random opponent
python chess_game.py --self-consistency --random-opponent --save-pgn
```

### Command Line Options

- `--model`: OpenAI model to use (default: gpt-3.5-turbo-instruct)
- `--stockfish`: Path to Stockfish executable (default: stockfish)
- `--time`: Time limit for Stockfish moves in seconds (default: 1.0)
- `--skill`: Stockfish skill level 0-20 (default: 5, where 20 is maximum strength)
- `--self-consistency`: Use self-consistency approach instead of single model
- `--random-opponent`: Use random legal moves instead of Stockfish
- `--model-color`: Color for the model (white/black, default: white)
- `--save-json`: Save game result as JSON file
- `--save-pgn`: Save game as PGN file
- `--no-save`: Don't save any files

### Stockfish Skill Levels

- **0-3**: Beginner level (good for testing, makes obvious mistakes)
- **4-7**: Intermediate level (recommended for LLMs, balanced play)
- **8-12**: Advanced level (strong tactical play)
- **13-17**: Expert level (very strong, few mistakes)
- **18-20**: Maximum strength (grandmaster level, near-perfect play)

**Default**: Skill level 5 (intermediate) - provides a good challenge for LLMs without being overwhelming.

### Examples

```bash
# Quick game with single model
python chess_game.py --save-pgn

# Self-consistency game with longer Stockfish thinking time
python chess_game.py --self-consistency --time 3.0 --save-json --save-pgn

# Model plays black with GPT-4
python chess_game.py --model gpt-4-turbo --model-color black --self-consistency
```

## Game Output

The engine provides real-time feedback:
- Current board position
- Move analysis
- Game progress
- Final result

## File Outputs

### JSON Format
Contains detailed game information:
- Move history with timestamps
- Player information
- Board states
- Game metadata
- **Fallback move tracking**: Records when and why fallback moves were used
- **Move quality indicators**: Shows which moves were AI-generated vs fallback

### PGN Format
Standard chess notation format that can be imported into chess software.

## Environment Setup

Make sure to set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in the parent directory with:
```
OPENAI_API_KEY=your-api-key-here
```

## Troubleshooting

### Stockfish Not Found
- Ensure Stockfish is installed and in your PATH
- Use `--stockfish /path/to/stockfish` to specify custom path

### API Errors
- Check your OpenAI API key is valid
- Ensure you have sufficient API credits
- Check rate limits

### Move Generation Issues
- The model may generate invalid moves occasionally
- Self-consistency approach is more robust but uses more tokens
- **Fallback System**: When the AI fails to generate a valid move, the engine automatically uses a random legal move
- Fallback moves are clearly marked in the output and game history
- Check the console output for error messages and fallback notifications

## Performance Notes

- **Single Model**: Faster, uses fewer tokens, but may be less accurate
- **Self-Consistency**: More robust, uses 3x tokens, better move quality
- **Stockfish Time**: Longer time = stronger play but slower games
- **Model Choice**: GPT-4 is stronger but more expensive than GPT-3.5

## Game Analysis

After playing games, you can:
- Import PGN files into chess analysis software
- Analyze JSON files for move patterns
- Compare single model vs self-consistency performance
- Study the model's chess understanding
