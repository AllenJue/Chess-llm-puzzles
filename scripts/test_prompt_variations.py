#!/usr/bin/env python3
"""
Test different prompt variations to find one that works for open source models.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dotenv import load_dotenv
from openai import OpenAI
import chess
import chess.pgn
from io import StringIO

# Load environment
env_file = os.path.join(parent_dir, '.env')
if os.path.exists(env_file):
    load_dotenv(env_file)
else:
    load_dotenv()

from main import read_chess_puzzles_csv

# Test model
MODEL = "meta-llama/llama-3.3-8b-instruct:free"

# Different prompt variations to test
PROMPT_VARIATIONS = [
    {
        "name": "Tactical focus",
        "system": """You are a chess grandmaster. Analyze the position and find the best move.
Look for checkmate, captures, and tactical opportunities.
Provide ONLY the next move in standard algebraic notation (SAN).
Example: Qe1+ or Nf6 or e4""",
    },
    {
        "name": "Checkmate emphasis",
        "system": """You are a chess grandmaster solving a chess puzzle.
Find the best move - look for checkmate, winning captures, or strong tactical moves.
Output format: Just the move in SAN notation.
Example responses: Qe1+ or Rxf7 or Nf6""",
    },
    {
        "name": "Best move only",
        "system": """Chess grandmaster. Find the best move in this position.
Output: Single move in SAN (e.g., Qe1+, Nf6, e4)""",
    },
    {
        "name": "Position analysis",
        "system": """You are analyzing a chess position. Find the strongest move.
Consider: checkmate threats, material gains, tactical opportunities.
Respond with only the move in standard algebraic notation.""",
    },
    {
        "name": "Puzzle solver",
        "system": """Solve this chess puzzle. Find the winning move.
Output format: <move>
Example: Qe1+""",
    },
    {
        "name": "Concise instruction",
        "system": """Best move in SAN. One word only.""",
    },
    {
        "name": "Checkmate finder",
        "system": """You are solving a chess puzzle. Look for checkmate moves first.
If there's a checkmate, play it. Otherwise find the best move.
Output: Just the move in SAN (e.g., Qe1#)""",
    },
    {
        "name": "Queen checkmate focus",
        "system": """Chess puzzle: Find checkmate. Look for queen moves that deliver checkmate.
Output format: <move> in SAN notation.""",
    },
    {
        "name": "FEN + puzzle",
        "system": """Analyze this chess position. Find the winning move.
The position is given in FEN notation. Look for checkmate opportunities.
Respond with only the move in standard algebraic notation.""",
    },
]


def get_puzzle_position(puzzle_idx=0):
    """Get the board position for puzzle 0."""
    df = read_chess_puzzles_csv(os.path.join(parent_dir, "data", "lichess_puzzles_with_pgn_1000.csv"))
    puzzle = df.iloc[puzzle_idx]
    
    # Parse the PGN to get the board position
    pgn_io = StringIO(puzzle['PGN_partial'])
    game = chess.pgn.read_game(pgn_io)
    
    # Replay to get the board
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    
    # Get the expected move
    moves = puzzle['Moves'].split()
    expected_move = moves[1] if len(moves) > 1 else None
    
    # Export current position as PGN
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    current_game = chess.pgn.Game.from_board(board)
    user_prompt = current_game.accept(exporter)
    
    return board, user_prompt, expected_move


def test_prompt_variation(variation, board, user_prompt, expected_move):
    """Test a prompt variation."""
    print(f"\n{'='*60}")
    print(f"Testing: {variation['name']}")
    print(f"{'='*60}")
    
    api_key = os.getenv("ANANNAS_API_KEY")
    base_url = os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    system_prompt = variation['system']
    
    # Add board FEN to user prompt for better context
    fen = board.fen()
    enhanced_user = f"{user_prompt}\n\nCurrent position (FEN): {fen}\nSide to move: {'White' if board.turn else 'Black'}"
    
    # Try with system message
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_user},
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=128,
            temperature=0.1,
        )
    except Exception as e:
        if "400" in str(e):
            # Fallback to combined
            combined = f"{system_prompt}\n\n{enhanced_user}"
            messages = [{"role": "user", "content": combined}]
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=128,
                temperature=0.1,
            )
        else:
            raise
    
    content = response.choices[0].message.content or ""
    print(f"Response: {content[:300]}")
    
    # Try to extract a move
    import re
    # Look for SAN patterns
    san_pattern = r'\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8]|[a-h][1-8]|O-O-O?|0-0-0?)\b'
    matches = re.findall(san_pattern, content, re.IGNORECASE)
    
    if matches:
        first_move = matches[0]
        print(f"Extracted move: {first_move}")
        
        # Try to convert to UCI
        try:
            move_obj = board.parse_san(first_move)
            uci = move_obj.uci()
            print(f"UCI: {uci}")
            
            # Check if it matches expected
            if expected_move:
                expected_uci = None
                try:
                    expected_move_obj = board.parse_san(expected_move)
                    expected_uci = expected_move_obj.uci()
                except:
                    pass
                
                if uci == expected_uci:
                    print(f"✅ CORRECT! Matches expected {expected_uci}")
                    return True
                else:
                    print(f"❌ Wrong. Expected: {expected_uci}, Got: {uci}")
        except Exception as e:
            print(f"❌ Invalid move: {e}")
    else:
        print("❌ No move found in response")
    
    return False


def main():
    print(f"Testing prompt variations with {MODEL}")
    print(f"Target: Solve puzzle 0\n")
    
    board, user_prompt, expected_move = get_puzzle_position(0)
    print(f"Board FEN: {board.fen()}")
    print(f"Expected move: {expected_move}")
    print(f"User prompt (first 200 chars): {user_prompt[:200]}...\n")
    
    results = {}
    for variation in PROMPT_VARIATIONS:
        try:
            success = test_prompt_variation(variation, board, user_prompt, expected_move)
            results[variation['name']] = success
            if success:
                print(f"\n🎉 SUCCESS with: {variation['name']}")
                break
        except Exception as e:
            print(f"❌ Error: {e}")
            results[variation['name']] = False
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}")


if __name__ == "__main__":
    main()

