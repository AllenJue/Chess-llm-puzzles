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
sys.path.append('MAD')
from utils.agent import Agent
from utils.openai_utils import num_tokens_from_string, model2max_context
import chess
import chess.pgn
import io

# Import new debate system
from chess_debate_v2 import ChessDebateV2, save_debate_history_v2


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


class ChessSelfConsistency:
    def __init__(self, 
                 model_name: str = 'gpt-3.5-turbo-instruct',
                 temperature: float = 0.1,
                 openai_api_key: str = None,
                 max_rounds: int = 2,
                 sleep_time: float = 0.1):
        """Create a chess self-consistency system"""
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

    def run_debate(self, user_prompt: str, expected_uci: str = None, played_plies: int = None, board_fen: str = None) -> tuple:
        """Run a chess self-consistency evaluation using 3 independent queries"""
        print(f"\n<self-consistency> : Starting self-consistency evaluation")
        
        # Build system prompt
        system_prompt = (
            "You are a chess grandmaster.\n"
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

        
        print(f"\n<debug> : Self-consistency system prompt building:")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        if expected_uci:
            print(f"Expected UCI: {expected_uci}")
        
        # Query 1-3: Independent analysis (3 independent queries)
        print(f"\n<self-consistency> : Running 3 independent queries")
        
        # Aggressive GM analysis
        print(f"\n<debug> : Aggressive GM - using query_model_for_move:")
        print(f"<debug> : agg_system_prompt: {repr(agg_system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        agg_response, aggressive_san, agg_token_info = self.aggressive_gm.model_interface.get_move_with_extraction(
            agg_system_prompt, user_prompt, 
            current_turn_number=played_plies // 2 + 1 if played_plies else None,
            is_white_to_move=(played_plies % 2 == 0) if played_plies else True
        )
        print(f"<debug> : Aggressive GM response: {repr(agg_response)}")
        print(f"<debug> : Aggressive GM move: {aggressive_san}")

        self.aggressive_gm.add_memory(agg_response)
        
        # Positional GM analysis
        print(f"\n<debug> : Positional GM - using query_model_for_move:")
        print(f"<debug> : pos_system_prompt: {repr(pos_system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        pos_response, positional_san, pos_token_info = self.positional_gm.model_interface.get_move_with_extraction(
            pos_system_prompt, user_prompt, 
            current_turn_number=played_plies // 2 + 1 if played_plies else None,
            is_white_to_move=(played_plies % 2 == 0) if played_plies else True
        )

        print(f"<debug> : Positional GM response: {repr(pos_response)}")
        print(f"<debug> : Positional GM move: {positional_san}")
        
        self.positional_gm.add_memory(f"Positional GM move: {positional_san}")
        
        # Neutral GM analysis
        print(f"\n<debug> : Neutral GM - using query_model_for_move:")
        print(f"<debug> : system_prompt: {repr(system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        neutral_response, neutral_san, neutral_token_info = self.neutral_gm.model_interface.get_move_with_extraction(
            system_prompt, user_prompt, 
            current_turn_number=played_plies // 2 + 1 if played_plies else None,
            is_white_to_move=(played_plies % 2 == 0) if played_plies else True
        )

        print(f"<debug> : Neutral GM response: {repr(neutral_response)}")
        print(f"<debug> : Neutral GM move: {neutral_san}")
        
        self.neutral_gm.add_memory(f"Neutral GM move: {neutral_san}")
        
        # Convert all moves to UCI
        agg_move_uci = san_to_uci(board_fen, aggressive_san) if aggressive_san else None
        pos_move_uci = san_to_uci(board_fen, positional_san) if positional_san else None
        neutral_move_uci = san_to_uci(board_fen, neutral_san) if neutral_san else None
        
        print(f"\n<self-consistency> : All UCI moves:")
        print(f"<self-consistency> : Aggressive GM: {agg_move_uci}")
        print(f"<self-consistency> : Positional GM: {pos_move_uci}")
        print(f"<self-consistency> : Neutral GM: {neutral_move_uci}")
        
        # Voting system: most frequent, then neutral, aggressive, positional
        all_moves = [agg_move_uci, pos_move_uci, neutral_move_uci]
        valid_moves = [move for move in all_moves if move is not None]
        
        if not valid_moves:
            self.final_move = None
            print(f"<self-consistency> : No valid moves extracted")
            return self.final_move, {}
        
        # Count frequency of moves
        from collections import Counter
        move_counts = Counter(valid_moves)
        most_frequent_moves = move_counts.most_common()
        
        print(f"\n<self-consistency> : Move frequency: {dict(move_counts)}")
        
        # Priority order: most frequent, then neutral, aggressive, positional
        priority_order = [neutral_move_uci, agg_move_uci, pos_move_uci]
        
        # First try most frequent move
        if most_frequent_moves:
            most_frequent_move = most_frequent_moves[0][0]
            if move_counts[most_frequent_move] > 1:  # If there's a majority
                print(f"<self-consistency> : Most frequent move: {most_frequent_move} (count: {move_counts[most_frequent_move]})")
                self.final_move = most_frequent_move
            else:
                # No consensus, use priority order
                for move in priority_order:
                    if move in valid_moves:
                        self.final_move = move
                        print(f"<self-consistency> : Using priority move: {move}")
                        break
        else:
            # Fallback to priority order
            for move in priority_order:
                if move in valid_moves:
                    self.final_move = move
                    print(f"<self-consistency> : Using fallback priority move: {move}")
                    break
        
        print(f"<self-consistency> : Final consensus move: {self.final_move}")
        
        # Collect self-consistency history
        self_consistency_history = {
            "query1": {
                "aggressive_move": aggressive_san,
                "aggressive_uci": agg_move_uci,
                "aggressive_response": agg_response,
                "aggressive_tokens": agg_token_info
            },
            "query2": {
                "positional_move": positional_san,
                "positional_uci": pos_move_uci,
                "positional_response": pos_response,
                "positional_tokens": pos_token_info
            },
            "query3": {
                "neutral_move": neutral_san,
                "neutral_uci": neutral_move_uci,
                "neutral_response": neutral_response,
                "neutral_tokens": neutral_token_info
            },
            "final_moves": {
                "aggressive_uci": agg_move_uci,
                "positional_uci": pos_move_uci,
                "neutral_uci": neutral_move_uci,
                "consensus_move": self.final_move
            },
            "total_tokens": {
                "aggressive": agg_token_info,
                "positional": pos_token_info,
                "neutral": neutral_token_info,
                "total_prompt_tokens": (agg_token_info.get("prompt_tokens", 0) if agg_token_info else 0) + 
                                      (pos_token_info.get("prompt_tokens", 0) if pos_token_info else 0) + 
                                      (neutral_token_info.get("prompt_tokens", 0) if neutral_token_info else 0),
                "total_completion_tokens": (agg_token_info.get("completion_tokens", 0) if agg_token_info else 0) + 
                                         (pos_token_info.get("completion_tokens", 0) if pos_token_info else 0) + 
                                         (neutral_token_info.get("completion_tokens", 0) if neutral_token_info else 0),
                "total_tokens": (agg_token_info.get("total_tokens", 0) if agg_token_info else 0) + 
                               (pos_token_info.get("total_tokens", 0) if pos_token_info else 0) + 
                               (neutral_token_info.get("total_tokens", 0) if neutral_token_info else 0)
            }
        }
        
        return self.final_move, self_consistency_history


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
                    debate: ChessSelfConsistency = None, debate_v2: ChessDebateV2 = None, 
                    max_puzzles: int = 5, start_puzzle: int = 0) -> pd.DataFrame:
    """
    Evaluate puzzles using either model interface, self-consistency system, or debate system.
    Faithful to the provided evaluation logic:
    - Automatically skips the first move after PGN_partial (the puzzle start move)
    - Tests every other move (model's turns)
    - Automatically plays the intermediate opponent moves
    - Uses correct ply alignment and alternates between model/reference turns
    
    Args:
        df: DataFrame with puzzle data (must have 'PGN_partial' and 'Moves' columns)
        model_interface: ChessModelInterface instance (for single model)
        debate: ChessSelfConsistency instance (for self-consistency mode)
        debate_v2: ChessDebateV2 instance (for debate mode with moderator/judge)
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
    df_eval["single_model_prompt_tokens"] = 0
    df_eval["single_model_completion_tokens"] = 0
    df_eval["single_model_total_tokens"] = 0
    df_eval["single_model_model"] = ""
    df_eval["single_model_finish_reason"] = ""
    # New V2 debate columns
    df_eval["moderator_decision"] = ""
    df_eval["judge_decision"] = ""
    df_eval["debate_v2_history"] = ""
    
    # Comprehensive token and cost tracking
    df_eval["aggressive_prompt_tokens"] = 0
    df_eval["aggressive_completion_tokens"] = 0
    df_eval["aggressive_total_tokens"] = 0
    df_eval["aggressive_model"] = ""
    df_eval["aggressive_finish_reason"] = ""
    df_eval["aggressive_response"] = ""
    
    df_eval["positional_prompt_tokens"] = 0
    df_eval["positional_completion_tokens"] = 0
    df_eval["positional_total_tokens"] = 0
    df_eval["positional_model"] = ""
    df_eval["positional_finish_reason"] = ""
    df_eval["positional_response"] = ""
    
    df_eval["neutral_prompt_tokens"] = 0
    df_eval["neutral_completion_tokens"] = 0
    df_eval["neutral_total_tokens"] = 0
    df_eval["neutral_model"] = ""
    df_eval["neutral_finish_reason"] = ""
    df_eval["neutral_response"] = ""
    
    df_eval["moderator_prompt_tokens"] = 0
    df_eval["moderator_completion_tokens"] = 0
    df_eval["moderator_total_tokens"] = 0
    df_eval["moderator_model"] = ""
    df_eval["moderator_finish_reason"] = ""
    df_eval["moderator_response"] = ""
    
    df_eval["judge_prompt_tokens"] = 0
    df_eval["judge_completion_tokens"] = 0
    df_eval["judge_total_tokens"] = 0
    df_eval["judge_model"] = ""
    df_eval["judge_finish_reason"] = ""
    df_eval["judge_response"] = ""
    
    df_eval["total_prompt_tokens"] = 0
    df_eval["total_completion_tokens"] = 0
    df_eval["total_tokens"] = 0
    df_eval["estimated_cost_usd"] = 0.0
    
    # Debate process information
    df_eval["debate_rounds"] = 0
    df_eval["early_consensus"] = False
    df_eval["single_model_fallback"] = False
    df_eval["both_models_failed"] = False
    df_eval["debate_success"] = False
    df_eval["final_reason"] = ""
    df_eval["supported_side"] = ""
    
    # Input information
    df_eval["board_fen"] = ""
    df_eval["played_plies"] = 0
    df_eval["current_turn"] = 0
    df_eval["is_white_to_move"] = True
    df_eval["user_prompt"] = ""
    df_eval["system_prompt"] = ""

    total_moves = 0
    total_correct_moves = 0
    puzzles_solved = 0

    # Select the range of puzzles to evaluate
    puzzle_range = df_eval.iloc[start_puzzle:start_puzzle + max_puzzles]
    for idx, row in puzzle_range.iterrows():
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
                        # Use old debate system
                        predicted_uci, debate_history = debate.run_debate(user_prompt, expected_uci, played_plies, current_board.fen())
                        print(f"Debate predicted UCI: {predicted_uci}")
                        
                        # Save self-consistency data to DataFrame
                        df_eval.loc[idx, "aggressive_move"] = debate_history["query1"]["aggressive_move"]
                        df_eval.loc[idx, "positional_move"] = debate_history["query2"]["positional_move"]
                        df_eval.loc[idx, "neutral_move"] = debate_history["query3"]["neutral_move"]
                        df_eval.loc[idx, "final_consensus_move"] = debate_history["final_moves"]["consensus_move"]
                        df_eval.loc[idx, "debate_history"] = str(debate_history)
                        
                        # Save token information for self-consistency
                        if "total_tokens" in debate_history:
                            token_info = debate_history["total_tokens"]
                            
                            # Aggressive tokens
                            if token_info.get("aggressive"):
                                aff_tokens = token_info["aggressive"]
                                df_eval.loc[idx, "aggressive_prompt_tokens"] = aff_tokens.get("prompt_tokens", 0)
                                df_eval.loc[idx, "aggressive_completion_tokens"] = aff_tokens.get("completion_tokens", 0)
                                df_eval.loc[idx, "aggressive_total_tokens"] = aff_tokens.get("total_tokens", 0)
                                df_eval.loc[idx, "aggressive_model"] = aff_tokens.get("model", "")
                                df_eval.loc[idx, "aggressive_finish_reason"] = aff_tokens.get("finish_reason", "")
                            
                            # Positional tokens
                            if token_info.get("positional"):
                                pos_tokens = token_info["positional"]
                                df_eval.loc[idx, "positional_prompt_tokens"] = pos_tokens.get("prompt_tokens", 0)
                                df_eval.loc[idx, "positional_completion_tokens"] = pos_tokens.get("completion_tokens", 0)
                                df_eval.loc[idx, "positional_total_tokens"] = pos_tokens.get("total_tokens", 0)
                                df_eval.loc[idx, "positional_model"] = pos_tokens.get("model", "")
                                df_eval.loc[idx, "positional_finish_reason"] = pos_tokens.get("finish_reason", "")
                            
                            # Neutral tokens
                            if token_info.get("neutral"):
                                neu_tokens = token_info["neutral"]
                                df_eval.loc[idx, "neutral_prompt_tokens"] = neu_tokens.get("prompt_tokens", 0)
                                df_eval.loc[idx, "neutral_completion_tokens"] = neu_tokens.get("completion_tokens", 0)
                                df_eval.loc[idx, "neutral_total_tokens"] = neu_tokens.get("total_tokens", 0)
                                df_eval.loc[idx, "neutral_model"] = neu_tokens.get("model", "")
                                df_eval.loc[idx, "neutral_finish_reason"] = neu_tokens.get("finish_reason", "")
                            
                            # Total tokens
                            df_eval.loc[idx, "total_prompt_tokens"] = token_info.get("total_prompt_tokens", 0)
                            df_eval.loc[idx, "total_completion_tokens"] = token_info.get("total_completion_tokens", 0)
                            df_eval.loc[idx, "total_tokens"] = token_info.get("total_tokens", 0)
                            
                            # Cost calculation will be done later
                            df_eval.loc[idx, "estimated_cost_usd"] = 0.0
                        
                        # Save detailed debate history to JSON file
                        save_debate_history(debate_history, idx)
                    elif debate_v2:
                        # Use new debate system with moderator and judge
                        predicted_uci, debate_history = debate_v2.run_debate(user_prompt, expected_uci, played_plies, current_board.fen())
                        print(f"Debate V2 predicted UCI: {predicted_uci}")
                        
                        # Save comprehensive debate V2 data to DataFrame
                        df_eval.loc[idx, "aggressive_move"] = debate_history["round1"]["affirmative_move"]
                        df_eval.loc[idx, "positional_move"] = debate_history["round1"]["negative_move"]
                        df_eval.loc[idx, "moderator_decision"] = str(debate_history["round1"]["moderator_response"])
                        df_eval.loc[idx, "judge_decision"] = debate_history["final_result"]["reason"]
                        df_eval.loc[idx, "final_consensus_move"] = debate_history["final_result"]["final_move_uci"]
                        df_eval.loc[idx, "debate_v2_history"] = str(debate_history)
                        
                        # Token information
                        if "total_tokens" in debate_history:
                            token_info = debate_history["total_tokens"]
                            
                            # Aggressive tokens
                            if token_info.get("affirmative"):
                                aff_tokens = token_info["affirmative"]
                                df_eval.loc[idx, "aggressive_prompt_tokens"] = aff_tokens.get("prompt_tokens", 0)
                                df_eval.loc[idx, "aggressive_completion_tokens"] = aff_tokens.get("completion_tokens", 0)
                                df_eval.loc[idx, "aggressive_total_tokens"] = aff_tokens.get("total_tokens", 0)
                                df_eval.loc[idx, "aggressive_model"] = aff_tokens.get("model", "")
                                df_eval.loc[idx, "aggressive_finish_reason"] = aff_tokens.get("finish_reason", "")
                            
                            # Positional tokens
                            if token_info.get("negative"):
                                neg_tokens = token_info["negative"]
                                df_eval.loc[idx, "positional_prompt_tokens"] = neg_tokens.get("prompt_tokens", 0)
                                df_eval.loc[idx, "positional_completion_tokens"] = neg_tokens.get("completion_tokens", 0)
                                df_eval.loc[idx, "positional_total_tokens"] = neg_tokens.get("total_tokens", 0)
                                df_eval.loc[idx, "positional_model"] = neg_tokens.get("model", "")
                                df_eval.loc[idx, "positional_finish_reason"] = neg_tokens.get("finish_reason", "")
                            
                            # Moderator tokens
                            if token_info.get("moderator"):
                                mod_tokens = token_info["moderator"]
                                df_eval.loc[idx, "moderator_prompt_tokens"] = mod_tokens.get("prompt_tokens", 0)
                                df_eval.loc[idx, "moderator_completion_tokens"] = mod_tokens.get("completion_tokens", 0)
                                df_eval.loc[idx, "moderator_total_tokens"] = mod_tokens.get("total_tokens", 0)
                                df_eval.loc[idx, "moderator_model"] = mod_tokens.get("model", "")
                                df_eval.loc[idx, "moderator_finish_reason"] = mod_tokens.get("finish_reason", "")
                            
                            # Judge tokens
                            if token_info.get("judge"):
                                judge_tokens = token_info["judge"]
                                df_eval.loc[idx, "judge_prompt_tokens"] = judge_tokens.get("prompt_tokens", 0)
                                df_eval.loc[idx, "judge_completion_tokens"] = judge_tokens.get("completion_tokens", 0)
                                df_eval.loc[idx, "judge_total_tokens"] = judge_tokens.get("total_tokens", 0)
                                df_eval.loc[idx, "judge_model"] = judge_tokens.get("model", "")
                                df_eval.loc[idx, "judge_finish_reason"] = judge_tokens.get("finish_reason", "")
                            
                            # Total tokens
                            df_eval.loc[idx, "total_prompt_tokens"] = token_info.get("total_prompt_tokens", 0)
                            df_eval.loc[idx, "total_completion_tokens"] = token_info.get("total_completion_tokens", 0)
                            df_eval.loc[idx, "total_tokens"] = token_info.get("total_tokens", 0)
                        
                        # Response information
                        df_eval.loc[idx, "aggressive_response"] = debate_history["round1"]["affirmative_response"]
                        df_eval.loc[idx, "positional_response"] = debate_history["round1"]["negative_response"]
                        df_eval.loc[idx, "moderator_response"] = str(debate_history["round1"]["moderator_response"])
                        
                        # Debate process information
                        df_eval.loc[idx, "early_consensus"] = debate_history["round1"].get("early_consensus", False)
                        df_eval.loc[idx, "single_model_fallback"] = debate_history["round1"].get("single_model", False)
                        df_eval.loc[idx, "both_models_failed"] = debate_history["round1"].get("failure", False)
                        df_eval.loc[idx, "debate_success"] = debate_history["final_result"]["success"]
                        df_eval.loc[idx, "final_reason"] = debate_history["final_result"]["reason"]
                        df_eval.loc[idx, "supported_side"] = debate_history["final_result"]["supported_side"]
                        
                        # Input information
                        df_eval.loc[idx, "board_fen"] = current_board.fen()
                        df_eval.loc[idx, "played_plies"] = played_plies
                        df_eval.loc[idx, "current_turn"] = played_plies // 2 + 1
                        df_eval.loc[idx, "is_white_to_move"] = current_board.turn
                        df_eval.loc[idx, "user_prompt"] = user_prompt
                        df_eval.loc[idx, "system_prompt"] = system_prompt
                        
                        # Estimate cost (rough calculation - adjust rates as needed)
                        total_tokens = df_eval.loc[idx, "total_tokens"]
                        # Cost calculation will be done later
                        df_eval.loc[idx, "estimated_cost_usd"] = 0.0
                        
                        # Save detailed debate history to JSON file
                        save_debate_history_v2(debate_history, idx)
                    else:
                        # Use single model
                        raw_response, predicted_san, token_info = model_interface.get_move_with_extraction(
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
                        
                        # Save token information for single model
                        if token_info:
                            df_eval.loc[idx, "single_model_prompt_tokens"] = token_info.get("prompt_tokens", 0)
                            df_eval.loc[idx, "single_model_completion_tokens"] = token_info.get("completion_tokens", 0)
                            df_eval.loc[idx, "single_model_total_tokens"] = token_info.get("total_tokens", 0)
                            df_eval.loc[idx, "single_model_model"] = token_info.get("model", "")
                            df_eval.loc[idx, "single_model_finish_reason"] = token_info.get("finish_reason", "")
                            
                            # Cost calculation will be done later
                            df_eval.loc[idx, "estimated_cost_usd"] = 0.0

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
            elif debate_v2:
                # Clear memory between puzzles
                debate_v2.clear_memory()
                
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
    parser.add_argument("--start-puzzle", type=int, default=0,
                       help="Starting puzzle index (0-based)")
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
    parser.add_argument("--self-consistency", action="store_true",
                       help="Use self-consistency system (3 independent queries with majority vote)")
    parser.add_argument("--debate", action="store_true",
                       help="Use multi-agent debate system with moderator and judge")
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
        if args.self_consistency:
            print(f"\nEvaluating puzzles with self-consistency system...")
            self_consistency = ChessSelfConsistency(
                model_name=args.model,
                temperature=0.1,
                openai_api_key=api_key,
                max_rounds=2,
                sleep_time=0.1
            )
            # Evaluate puzzles with self-consistency system
            df_results = evaluate_puzzles(df, debate=self_consistency, max_puzzles=args.max_puzzles, start_puzzle=args.start_puzzle)
        elif args.debate:
            print(f"\nEvaluating puzzles with new multi-agent debate system (moderator + judge)...")
            debate_v2 = ChessDebateV2(
                model_name=args.model,
                temperature=0.1,
                openai_api_key=api_key,
                max_rounds=3,
                sleep_time=0.1
            )
            # Evaluate puzzles with new debate system
            df_results = evaluate_puzzles(df, debate_v2=debate_v2, max_puzzles=args.max_puzzles, start_puzzle=args.start_puzzle)
        else:
            print(f"\nEvaluating puzzles with {args.model}...")
            model_interface = ChessModelInterface(api_key=api_key, model_name=args.model)
            # Evaluate puzzles with single model
            df_results = evaluate_puzzles(df, model_interface=model_interface, max_puzzles=args.max_puzzles, start_puzzle=args.start_puzzle)
        
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
