"""
Main Entry Point for Chess Puzzle Evaluation

This script provides the main interface for evaluating chess puzzles using
OpenAI models and Glicko-2 rating system.

Usage:
    python main.py --help
    python main.py --csv-file puzzles.csv --max-puzzles 100
    python main.py --evaluate --model gpt-4-turbo
"""

import argparse
import os
import sys
import json
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# Import our modules
from csv_reader import read_chess_puzzles_csv, sample_puzzles, save_puzzles_csv, get_puzzle_stats
from model_interface import ChessModelInterface
from chess_utils import build_chess_prompts, get_partial_pgn_from_url, extract_predicted_move, san_to_uci
from glicko_rating import update_agent_rating_from_puzzles, Rating

# Import debate functionality
import sys
sys.path.append('Multi-Agents-Debate/code')
from utils.agent import Agent
from utils.openai_utils import num_tokens_from_string, model2max_context
import chess
import chess.pgn
import io


class ChessDebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float = 0) -> None:
        """Create a chess debate player"""
        super(ChessDebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key
        self.model_interface = ChessModelInterface(api_key=openai_api_key, model_name=model_name)

    def ask(self, temperature: float=None):
        """Query for answer using our model interface instead of the old API"""
        # Build combined prompt from memory
        combined_prompt = ""
        for msg in self.memory_lst:
            if msg["role"] == "system":
                combined_prompt += msg["content"] + "\n"
            elif msg["role"] == "user":
                combined_prompt += msg["content"] + "\n"
            elif msg["role"] == "assistant":
                combined_prompt += msg["content"] + "\n"
        
        print(f"\n<debug> : ChessDebatePlayer '{self.name}' ask() method:")
        print(f"<debug> : combined_prompt: {repr(combined_prompt[:200])}...")
        
        # Use our model interface instead of the old API
        response = self.model_interface.query_model_for_move(
            system_prompt="",  # Already included in combined_prompt
            user_prompt=combined_prompt,
            max_tokens=500,
            temperature=temperature if temperature else self.temperature,
            top_p=1
        )
        
        print(f"<debug> : response: {repr(response)}")
        return response


class ChessDebate:
    def __init__(self, 
                 model_name: str = 'gpt-3.5-turbo-instruct',
                 temperature: float = 0.1,
                 openai_api_key: str = None,
                 max_rounds: int = 2,
                 sleep_time: float = 0.1):
        """Create a chess debate"""
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.max_rounds = max_rounds
        self.sleep_time = sleep_time
        
        # Initialize players
        self.aggressive_gm = ChessDebatePlayer(
            model_name=model_name,
            name="Mikhail Tal (Aggressive)",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time
        )
        
        self.positional_gm = ChessDebatePlayer(
            model_name=model_name,
            name="Magnus Carlsen (Positional)",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time
        )
        
        self.neutral_gm = ChessDebatePlayer(
            model_name=model_name,
            name="Neutral GM",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time
        )
        
        # Set up personas
        self.setup_personas()
        
        # Debate results
        self.debate_history = []
        self.final_move = None

    def setup_personas(self):
        """Set up the grandmaster personas"""
        # Aggressive GM (Mikhail Tal) - focuses on tactics and sacrifices
#         aggressive_prompt = """You are an aggressive chess grandmaster, with a style similar to Mikhail Tal.
# You will be given a partially completed game.
# After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.
# Use standard algebraic notation, e.g. "e4" or "Rdf8" or "R1a3".
# ALWAYS repeat the entire representation of the game so far.
# NEVER explain your choice.
# """
        aggressive_prompt = """You are a chess grandmaster.
You will be given a partially completed game.
After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.
Use standard algebraic notation, e.g. "e4" or "Rdf8" or "R1a3".
ALWAYS repeat the entire representation of the game so far.
NEVER explain your choice.
"""
        
        # Positional GM (Magnus Carlsen) - focuses on positional understanding
#         positional_prompt = """You are a positional chess grandmaster, similar to Magnus Carlsen.
# You will be given a partially completed game.
# After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.
# Use standard algebraic notation, e.g. "e4" or "Rdf8" or "R1a3".
# ALWAYS repeat the entire representation of the game so far.
# NEVER explain your choice."""

        positional_prompt = """You are a chess grandmaster.
  You will be given a partially completed game.
  After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.
  Use standard algebraic notation, e.g. "e4" or "Rdf8" or "R1a3".
  ALWAYS repeat the entire representation of the game so far.
  NEVER explain your choice.
  """
        
        
        # Neutral GM uses default prompt (no special persona)
        neutral_prompt = """You are a chess grandmaster.
You will be given a partially completed game.
After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.
Use standard algebraic notation, e.g. "e4" or "Rdf8" or "R1a3".
ALWAYS repeat the entire representation of the game so far.
NEVER explain your choice.
"""
        
        self.aggressive_gm.set_meta_prompt(aggressive_prompt)
        self.positional_gm.set_meta_prompt(positional_prompt)
        self.neutral_gm.set_meta_prompt(neutral_prompt)

    def get_board_fen_from_pgn(self, pgn_string: str) -> str:
        """Get the board FEN from a PGN string"""
        try:
            # Parse the PGN using io.StringIO
            pgn_io = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                return chess.Board().fen()
            
            # Replay the game to get the final position
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            
            return board.fen()
        except Exception as e:
            print(f"Error parsing PGN: {e}")
            return chess.Board().fen()

    def clear_memory(self):
        """Clear memory between puzzles to prevent token buildup"""
        self.aggressive_gm.memory_lst = []
        self.positional_gm.memory_lst = []
        self.neutral_gm.memory_lst = []
        # Re-setup personas after clearing
        self.setup_personas()

    # This is more self consistency rn. 
    def run_debate(self, user_prompt: str, expected_uci: str = None, played_plies: int = None, board_fen: str = None) -> tuple:
        """Run a chess debate on a position using multi-agent debate structure"""
        print(f"\n<debate> : Starting chess debate")
        
        # Build system prompt
        system_prompt = (
            "You are an chess grandmaster.\n"
            "You will be given a partially completed game.\n"
            "After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.\n"
            "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'."
        )

        agg_system_prompt = (
            "You are an aggressive chess grandmaster.\n"
            "You will be given a partially completed game.\n"
            "After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.\n"
            "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'."
        )

        pos_system_prompt = (
            "You are a positional chess grandmaster.\n"
            "You will be given a partially completed game.\n"
            "After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.\n"
            "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'."
        )

        
        print(f"\n<debug> : Debate system prompt building:")
        # print(f"<debug> : system_prompt: {repr(system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        if expected_uci:
            print(f"Expected UCI: {expected_uci}")
        
        # Round 1: Initial analysis (following multi-agent debate structure)
        print(f"\n<debate> : Round 1 - Initial analysis")
        
        # Aggressive GM analysis - use query_model_for_move for Round 1
        print(f"\n<debug> : Aggressive GM Round 1 - using query_model_for_move:")
        # print(f"<debug> : system_prompt: {repr(system_prompt)}")
        print(f"<debug> : agg_system_prompt: {repr(agg_system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        agg_response, aggressive_san = self.aggressive_gm.model_interface.get_move_with_extraction(
            agg_system_prompt, user_prompt, 
            current_turn_number=played_plies // 2 + 1 if played_plies else None,
            is_white_to_move=(played_plies % 2 == 0) if played_plies else True
        )
        print(f"<debug> : Aggressive GM Round 1 response: {repr(agg_response)}")
        print(f"<debug> : Aggressive GM Round 1 move: {aggressive_san}")

        self.aggressive_gm.add_memory(agg_response)
        
        # Positional GM analysis - use query_model_for_move for Round 1
        print(f"\n<debug> : Positional GM Round 1 - using query_model_for_move:")
        # print(f"<debug> : system_prompt: {repr(system_prompt)}")
        print(f"<debug> : pos_system_prompt: {repr(pos_system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        pos_response, positional_san = self.positional_gm.model_interface.get_move_with_extraction(
            pos_system_prompt, user_prompt, 
            current_turn_number=played_plies // 2 + 1 if played_plies else None,
            is_white_to_move=(played_plies % 2 == 0) if played_plies else True
        )

        print(f"<debug> : Positional GM Round 1 response: {repr(pos_response)}")
        print(f"<debug> : Positional GM Round 1 move: {positional_san}")
        
        self.positional_gm.add_memory(f"Positional GM move: {positional_san}")
        
        # Neutral GM analysis - use default system prompt for Round 1
        print(f"\n<debug> : Neutral GM Round 1 - using query_model_for_move:")
        print(f"<debug> : system_prompt: {repr(system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        neutral_response, neutral_san = self.neutral_gm.model_interface.get_move_with_extraction(
            system_prompt, user_prompt, 
            current_turn_number=played_plies // 2 + 1 if played_plies else None,
            is_white_to_move=(played_plies % 2 == 0) if played_plies else True
        )

        print(f"<debug> : Neutral GM Round 1 response: {repr(neutral_response)}")
        print(f"<debug> : Neutral GM Round 1 move: {neutral_san}")
        
        self.neutral_gm.add_memory(f"Neutral GM move: {neutral_san}")
        
        # Round 2: Explanation and Debate
        print(f"\n<debate> : Round 2 - Explanation and Debate")
        
        # Aggressive GM explains and debates
        self.aggressive_gm.add_event(f"""
The positional GM suggests: {positional_san}
The neutral GM suggests: {neutral_san}
Your initial suggestion was: {aggressive_san}

Now succinctly explain your reasoning and debate which move is better.
Consider tactical vs positional factors, and defend your choice.
Why is your move superior to the other GMs' suggestions?
""")
        agg_explanation = self.aggressive_gm.ask()
        self.aggressive_gm.add_memory(agg_explanation)
        
        # Positional GM explains and debates
        self.positional_gm.add_event(f"""
The aggressive GM suggests: {aggressive_san}
The neutral GM suggests: {neutral_san}
Your initial suggestion was: {positional_san}

Now succinctly explain your reasoning and debate which move is better.
Consider tactical vs positional factors, and defend your choice.
Why is your move superior to the other GMs' suggestions?
""")
        pos_explanation = self.positional_gm.ask()
        self.positional_gm.add_memory(pos_explanation)
        
        # Neutral GM explains and debates
        self.neutral_gm.add_event(f"""
The aggressive GM suggests: {aggressive_san}
The positional GM suggests: {positional_san}
Your initial suggestion was: {neutral_san}

Now succinctly explain your reasoning and debate which move is better.
Consider tactical vs positional factors, and defend your choice.
Why is your move superior to the other GMs' suggestions?
""")
        neutral_explanation = self.neutral_gm.ask()
        self.neutral_gm.add_memory(neutral_explanation)
        
        # Round 3: Final Consensus
        print(f"\n<debate> : Round 3 - Final Consensus")
        
        # Aggressive GM final recommendation
        self.aggressive_gm.add_event(f"""
Based on the debate, provide your FINAL recommendation.
Give ONLY the move in standard algebraic notation (e.g., "Qe1#" or "Nf3").
Do not explain further - just the move.
""")
        final_agg = self.aggressive_gm.ask()
        self.aggressive_gm.add_memory(final_agg)
        
        # Positional GM final recommendation
        self.positional_gm.add_event(f"""
Based on the debate, provide your FINAL recommendation.
Give ONLY the move in standard algebraic notation (e.g., "Qe1#" or "Nf3").
Do not explain further - just the move.
""")
        final_pos = self.positional_gm.ask()
        self.positional_gm.add_memory(final_pos)
        
        # Neutral GM final recommendation
        self.neutral_gm.add_event(f"""
Based on the debate, provide your FINAL recommendation.
Give ONLY the move in standard algebraic notation (e.g., "Qe1#" or "Nf3").
Do not explain further - just the move.
""")
        final_neutral = self.neutral_gm.ask()
        self.neutral_gm.add_memory(final_neutral)
        
        # Extract moves from all three GMs
        agg_move_san = extract_predicted_move(final_agg)
        pos_move_san = extract_predicted_move(final_pos)
        neutral_move_san = extract_predicted_move(final_neutral)
        
        # Print all moves for debugging
        print(f"\n<debate> : All final moves:")
        print(f"<debate> : Aggressive GM: {agg_move_san}")
        print(f"<debate> : Positional GM: {pos_move_san}")
        print(f"<debate> : Neutral GM: {neutral_move_san}")
        
        # Convert all moves to UCI
        agg_move_uci = san_to_uci(board_fen, agg_move_san) if agg_move_san else None
        pos_move_uci = san_to_uci(board_fen, pos_move_san) if pos_move_san else None
        neutral_move_uci = san_to_uci(board_fen, neutral_move_san) if neutral_move_san else None
        
        print(f"\n<debate> : All UCI moves:")
        print(f"<debate> : Aggressive GM: {agg_move_uci}")
        print(f"<debate> : Positional GM: {pos_move_uci}")
        print(f"<debate> : Neutral GM: {neutral_move_uci}")
        
        # Voting system: most frequent, then neutral, aggressive, positional
        all_moves = [agg_move_uci, pos_move_uci, neutral_move_uci]
        valid_moves = [move for move in all_moves if move is not None]
        
        if not valid_moves:
            self.final_move = None
            print(f"<debate> : No valid moves extracted")
            return self.final_move
        
        # Count frequency of moves
        from collections import Counter
        move_counts = Counter(valid_moves)
        most_frequent_moves = move_counts.most_common()
        
        print(f"\n<debate> : Move frequency: {dict(move_counts)}")
        
        # Priority order: most frequent, then neutral, aggressive, positional
        priority_order = [neutral_move_uci, agg_move_uci, pos_move_uci]
        
        # First try most frequent move
        if most_frequent_moves:
            most_frequent_move = most_frequent_moves[0][0]
            if move_counts[most_frequent_move] > 1:  # If there's a tie, use priority order
                print(f"<debate> : Most frequent move: {most_frequent_move} (count: {move_counts[most_frequent_move]})")
                self.final_move = most_frequent_move
            else:
                # No consensus, use priority order
                for move in priority_order:
                    if move in valid_moves:
                        self.final_move = move
                        print(f"<debate> : Using priority move: {move}")
                        break
        else:
            # Fallback to priority order
            for move in priority_order:
                if move in valid_moves:
                    self.final_move = move
                    print(f"<debate> : Using fallback priority move: {move}")
                    break
        
        print(f"<debate> : Final consensus move: {self.final_move}")
        
        # Collect debate history
        debate_history = {
            "round1": {
                "aggressive_move": aggressive_san,
                "positional_move": positional_san,
                "neutral_move": neutral_san,
                "aggressive_response": agg_response,
                "positional_response": pos_response,
                "neutral_response": neutral_response
            },
            "round2": {
                "aggressive_explanation": agg_explanation,
                "positional_explanation": pos_explanation,
                "neutral_explanation": neutral_explanation
            },
            "round3": {
                "aggressive_final": final_agg,
                "positional_final": final_pos,
                "neutral_final": final_neutral
            },
            "final_moves": {
                "aggressive_uci": agg_move_uci,
                "positional_uci": pos_move_uci,
                "neutral_uci": neutral_move_uci,
                "consensus_move": self.final_move
            }
        }
        
        return self.final_move, debate_history


def save_debate_history(debate_history, puzzle_idx, output_dir="debate_history"):
    """Save debate history to a JSON file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"puzzle_{puzzle_idx}_debate.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(debate_history, f, indent=2)
    
    print(f"Debate history saved to {filepath}")


def load_environment():
    """Load environment variables from .env file if it exists."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
    else:
        print("No .env file found, using system environment variables")


def evaluate_puzzles(df: pd.DataFrame, model_interface: ChessModelInterface = None, 
                    debate: ChessDebate = None, max_puzzles: int = 5) -> pd.DataFrame:
    """
    Evaluate puzzles using either model interface or debate system.
    Faithful to the provided evaluation logic:
    - Automatically skips the first move after PGN_partial (the puzzle start move)
    - Tests every other move (model's turns)
    - Automatically plays the intermediate opponent moves
    - Uses correct ply alignment and alternates between model/reference turns
    
    Args:
        df: DataFrame with puzzle data (must have 'PGN_partial' and 'Moves' columns)
        model_interface: ChessModelInterface instance (for single model)
        debate: ChessDebate instance (for debate mode)
        max_puzzles: Maximum number of puzzles to evaluate
        
    Returns:
        DataFrame with evaluation results
    """
    df_eval = df.copy()
    df_eval["correct_moves"] = 0
    df_eval["puzzle_solved"] = False
    df_eval["error"] = ""
    df_eval["aggressive_move"] = ""
    df_eval["positional_move"] = ""
    df_eval["neutral_move"] = ""
    df_eval["final_consensus_move"] = ""
    df_eval["debate_history"] = ""
    df_eval["single_model_response"] = ""
    df_eval["single_model_move"] = ""

    total_moves = 0
    total_correct_moves = 0
    puzzles_solved = 0

    for idx, row in df_eval.head(max_puzzles).iterrows():
        print(f"\n=== Evaluating puzzle {idx} ===")

        try:
            # --- Build starting board from PGN partial
            pgn_io = io.StringIO(row["PGN_partial"])
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                print("Invalid PGN_partial -- skipping")
                df_eval.loc[idx, "error"] = "Invalid PGN_partial"
                continue
                
            board = game.board()
            played_plies = 0
            for mv in game.mainline_moves():
                board.push(mv)
                played_plies += 1

            solution_moves = row["Moves"].split() if pd.notna(row["Moves"]) else []
            if len(solution_moves) == 0:
                print("No moves to play in solution")
                df_eval.loc[idx, "error"] = "No moves to play in solution"
                continue

            current_board = board.copy()
            initial_model_side = current_board.turn
            correct_for_puzzle = 0
            error = ""

            ply = 1  # start from the second solution move (first is automatically played in PGN)

            while ply < len(solution_moves):
                expected_uci = solution_moves[ply]

                if current_board.turn == initial_model_side:
                    # Model's turn to play
                    system_prompt = (
                        "You are a chess grandmaster.\n"
                        "You will be given a partially completed game.\n"
                        "After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.\n"
                        "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'."
                    )

                    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
                    current_game = chess.pgn.Game.from_board(current_board)
                    user_prompt = current_game.accept(exporter)

                    print("\n--- Model Turn ---")
                    print("Board:")
                    print(current_board)
                    print("Expected UCI:", expected_uci)

                    if debate:
                        # Use debate system
                        predicted_uci, debate_history = debate.run_debate(user_prompt, expected_uci, played_plies, current_board.fen())
                        print(f"Debate predicted UCI: {predicted_uci}")
                        
                        # Save debate data to DataFrame
                        df_eval.loc[idx, "aggressive_move"] = debate_history["round1"]["aggressive_move"]
                        df_eval.loc[idx, "positional_move"] = debate_history["round1"]["positional_move"]
                        df_eval.loc[idx, "neutral_move"] = debate_history["round1"]["neutral_move"]
                        df_eval.loc[idx, "final_consensus_move"] = debate_history["final_moves"]["consensus_move"]
                        df_eval.loc[idx, "debate_history"] = str(debate_history)
                        
                        # Save detailed debate history to JSON file
                        save_debate_history(debate_history, idx)
                    else:
                        # Use single model
                        raw_response, predicted_san = model_interface.get_move_with_extraction(
                            system_prompt, user_prompt,
                            current_turn_number=played_plies // 2 + 1,
                            is_white_to_move=current_board.turn
                        )
                        print("Predicted Response:", raw_response)
                        print("Turn number:", played_plies // 2 + 1)
                        print("Is white to move:", current_board.turn)
                        
                        if not predicted_san:
                            error = f"Failed to extract SAN at puzzle {idx}, ply {ply}"
                            print(f"❌ Failed to extract SAN at puzzle {idx}, ply {ply}")
                            break
                            
                        predicted_uci = san_to_uci(current_board.fen(), predicted_san)
                        print(f"Predicted UCI: {predicted_uci}")
                        
                        # Save single model data to DataFrame
                        df_eval.loc[idx, "single_model_response"] = raw_response
                        df_eval.loc[idx, "single_model_move"] = predicted_san

                    if predicted_uci == expected_uci:
                        print("✅ Correct move!")
                        current_board.push_uci(expected_uci)
                        correct_for_puzzle += 1
                        total_correct_moves += 1
                        total_moves += 1
                    else:
                        error = f"Mismatch: expected {expected_uci} but got {predicted_uci}"
                        print(f"❌ Mismatch: expected {expected_uci} but got {predicted_uci}")
                        break
                else:
                    # Opponent's turn, automatically play this move (no testing or scoring)
                    try:
                        current_board.push_uci(expected_uci)
                        print(f"Auto-played opponent move: {expected_uci}")
                    except Exception as e:
                        print(f"Error applying opponent move {expected_uci}: {e}")
                        error = f"Error applying opponent move {expected_uci}: {e}"
                        break

                ply += 1
                played_plies += 1

            df_eval.loc[idx, "correct_moves"] = correct_for_puzzle
            df_eval.loc[idx, "puzzle_solved"] = (correct_for_puzzle == (len(solution_moves) // 2))
            df_eval.loc[idx, "error"] = error
            
            # Number of moves model should play (half the moves after skipping the first)
            print("Model solved everything:", correct_for_puzzle == (len(solution_moves) // 2))
            
            if debate:
                # Clear memory between puzzles
                debate.clear_memory()
                
        except Exception as e:
            print(f"Error processing puzzle {idx}: {e}")
            df_eval.loc[idx, "error"] = str(e)
    
    return df_eval


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chess Puzzle Evaluator")
    
    parser.add_argument("--csv-file", default="lichess_puzzles_with_pgn_1000.csv",
                       help="Path to CSV file with puzzle data")
    parser.add_argument("--max-puzzles", type=int, default=10,
                       help="Maximum number of puzzles to evaluate")
    parser.add_argument("--model", choices=["gpt-3.5-turbo-instruct", "gpt-4-turbo"],
                       default="gpt-3.5-turbo-instruct", help="Model to use")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run puzzle evaluation")
    parser.add_argument("--sample", type=int, default=None,
                       help="Sample N puzzles randomly")
    parser.add_argument("--stats", action="store_true",
                       help="Show puzzle statistics")
    parser.add_argument("--rating", action="store_true",
                       help="Calculate Glicko-2 rating")
    parser.add_argument("--debate", action="store_true",
                       help="Use multi-agent debate system instead of single model")
    parser.add_argument("--output", default=None,
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Load environment
    load_environment()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in the .env file or environment")
        sys.exit(1)
    
    # Read CSV file
    try:
        df = read_chess_puzzles_csv(args.csv_file)
        print(f"Loaded {len(df)} puzzles from {args.csv_file}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Sample puzzles if requested
    if args.sample:
        df = sample_puzzles(df, n=args.sample)
        print(f"Sampled {len(df)} puzzles")
    
    # Show statistics
    if args.stats:
        stats = get_puzzle_stats(df)
        print("\nPuzzle Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Run evaluation
    if args.evaluate:
        if args.debate:
            print(f"\nEvaluating puzzles with multi-agent debate system...")
            debate = ChessDebate(
                model_name=args.model,
                temperature=0.1,
                openai_api_key=api_key,
                max_rounds=2,
                sleep_time=0.1
            )
            # Evaluate puzzles with debate
            df_results = evaluate_puzzles(df, debate=debate, max_puzzles=args.max_puzzles)
        else:
            print(f"\nEvaluating puzzles with {args.model}...")
            model_interface = ChessModelInterface(api_key=api_key, model_name=args.model)
            # Evaluate puzzles with single model
            df_results = evaluate_puzzles(df, model_interface=model_interface, max_puzzles=args.max_puzzles)
        
        # Save results
        if args.output:
            save_puzzles_csv(df_results, args.output)
            print(f"Results saved to {args.output}")
        
        # Show summary
        solved = df_results["puzzle_solved"].sum()
        total = len(df_results)
        print(f"\nEvaluation Summary:")
        print(f"  Puzzles solved: {solved}/{total} ({solved/total*100:.1f}%)")
        
        # Show individual results
        for idx, row in df_results.head(args.max_puzzles).iterrows():
            if row["error"] == "":
                status = "✓" if row["puzzle_solved"] else "✗"
                expected_moves = row["Moves"].split() if pd.notna(row["Moves"]) else []
                expected_move = expected_moves[1] if len(expected_moves) > 1 else "N/A"
                print(f"Puzzle {idx}: {status} Expected: {expected_move}, Correct moves: {row['correct_moves']}")
            else:
                print(f"Puzzle {idx}: ERROR - {row['error']}")
    
    # Calculate rating
    if args.rating:
        print("\nCalculating Glicko-2 rating...")
        if "puzzle_solved" in df.columns:
            agent_rating = update_agent_rating_from_puzzles(df)
            print(f"Agent rating: {agent_rating}")
        else:
            print("Error: No puzzle_solved column found. Run evaluation first.")


if __name__ == "__main__":
    main()
