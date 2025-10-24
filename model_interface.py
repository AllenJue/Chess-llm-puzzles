"""
Model Interface Module

This module handles interactions with OpenAI's API for chess move prediction.
It provides functions to query different models and extract chess moves from responses.
"""

import os
import time
from typing import Optional
from openai import OpenAI
from chess_utils import extract_predicted_move, san_to_uci


class ChessModelInterface:
    """
    Interface for interacting with OpenAI models for chess move prediction.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo-instruct"):
        """
        Initialize the model interface.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If None, will try to get from environment.
            model_name (str): Model name to use
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def query_model_for_move(self, system_prompt: str, user_prompt: str, 
                           max_tokens: int = 500, temperature: float = 0.1, 
                           top_p: float = 1) -> Optional[str]:
        """
        Call OpenAI instruct model with system and user prompts and return predicted move SAN.

        Args:
            system_prompt (str): The system prompt guiding the model.
            user_prompt (str): The user prompt with current game state.
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter

        Returns:
            Optional[str]: Predicted SAN move or None on failure.
        """
        try:
            combined_prompt = system_prompt + "\n" + user_prompt

            response = self.client.completions.create(
                model=self.model_name,
                prompt=combined_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            print(f"Model response: {response}")
            predicted_move_san = response.choices[0].text.strip()
            print("Model output:", repr(predicted_move_san))

            return predicted_move_san

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None

    def query_model_for_gpt4_move(self, system_prompt: str, user_prompt: str,
                                 max_tokens: int = 500, temperature: float = 0.1,
                                 top_p: float = 1) -> Optional[str]:
        """
        Call OpenAI GPT-4 model with system and user prompts and return predicted move SAN.

        Args:
            system_prompt (str): The system prompt guiding the model.
            user_prompt (str): The user prompt with current game state.
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter

        Returns:
            Optional[str]: Predicted SAN move or None on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",  # Use the gpt-4-turbo model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            print(f"Model response: {response}")

            # Extract the predicted move from the chat completion response
            if response.choices and response.choices[0].message:
                predicted_move_san = response.choices[0].message.content.strip()
                print("Model output:", repr(predicted_move_san))
                return predicted_move_san
            else:
                print("Model returned no response.")
                return None

        except Exception as e:
            print(f"Error calling OpenAI API with GPT-4: {e}")
            return None

    def get_move_with_extraction(self, system_prompt: str, user_prompt: str,
                               current_turn_number: Optional[int] = None,
                               is_white_to_move: bool = True,
                               use_gpt4: bool = False) -> tuple[Optional[str], Optional[str]]:
        """
        Get move from model and extract SAN move from response.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            current_turn_number (Optional[int]): Current turn number
            is_white_to_move (bool): Whether it's white's turn
            use_gpt4 (bool): Whether to use GPT-4 instead of instruct model
            
        Returns:
            tuple[Optional[str], Optional[str]]: (raw_response, extracted_san_move)
        """
        if use_gpt4:
            response_text = self.query_model_for_gpt4_move(system_prompt, user_prompt)
        else:
            response_text = self.query_model_for_move(system_prompt, user_prompt)
        
        if not response_text:
            return None, None
        
        extracted_san = extract_predicted_move(
            response_text=response_text,
            current_turn_number=current_turn_number,
            is_white_to_move=is_white_to_move
        )
        
        return response_text, extracted_san

    def get_uci_move(self, system_prompt: str, user_prompt: str, current_fen: str,
                    current_turn_number: Optional[int] = None,
                    is_white_to_move: bool = True,
                    use_gpt4: bool = False) -> Optional[str]:
        """
        Get UCI move from model.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            current_fen (str): Current board FEN
            current_turn_number (Optional[int]): Current turn number
            is_white_to_move (bool): Whether it's white's turn
            use_gpt4 (bool): Whether to use GPT-4
            
        Returns:
            Optional[str]: UCI move string
        """
        san_move = self.get_move_with_extraction(
            system_prompt, user_prompt, current_turn_number, is_white_to_move, use_gpt4
        )
        
        if not san_move:
            return None
        
        return san_to_uci(current_fen, san_move)


def query_model_for_move(system_prompt: str, user_prompt: str, api_key: str) -> Optional[str]:
    """
    Legacy function for backward compatibility.
    Call OpenAI instruct model with system and user prompts and return predicted move SAN.

    Args:
        system_prompt (str): The system prompt guiding the model.
        user_prompt (str): The user prompt with current game state.
        api_key (str): OpenAI API key.

    Returns:
        Optional[str]: Predicted SAN move or None on failure.
    """
    interface = ChessModelInterface(api_key=api_key)
    return interface.query_model_for_move(system_prompt, user_prompt)


def query_model_for_gpt4_move(system_prompt: str, user_prompt: str, api_key: str) -> Optional[str]:
    """
    Legacy function for backward compatibility.
    Call OpenAI GPT-4 model with system and user prompts and return predicted move SAN.

    Args:
        system_prompt (str): The system prompt guiding the model.
        user_prompt (str): The user prompt with current game state.
        api_key (str): OpenAI API key.

    Returns:
        Optional[str]: Predicted SAN move or None on failure.
    """
    interface = ChessModelInterface(api_key=api_key)
    return interface.query_model_for_gpt4_move(system_prompt, user_prompt)


def process_puzzles_with_model(df, model_interface: ChessModelInterface, 
                             max_puzzles: int = 5, api_delay: float = 0.1) -> None:
    """
    Process puzzles with rate limiting to avoid API limits.
    
    Args:
        df: DataFrame with puzzle data
        model_interface: ChessModelInterface instance
        max_puzzles (int): Maximum number of puzzles to process
        api_delay (float): Delay between API calls in seconds
    """
    for i, url in enumerate(df["GameUrl"].head(max_puzzles)):
        try:
            # Process puzzle here
            print(f"Processing puzzle {i+1}/{max_puzzles}")
            time.sleep(api_delay)
        except Exception as e:
            print(f"Error processing puzzle {i+1}: {e}")
            continue


if __name__ == "__main__":
    # Example usage
    print("Model Interface Module - Example Usage")
    print("=" * 40)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in the environment or .env file")
    else:
        print("OpenAI API key found")
        
        # Create model interface
        model_interface = ChessModelInterface(api_key=api_key)
        print(f"Model interface created with model: {model_interface.model_name}")
        
        # Example system and user prompts
        system_prompt = "You are a chess grandmaster. Provide the next move in standard algebraic notation."
        user_prompt = "1. e4 e5 2. Nf3"
        
        print("\nExample prompts:")
        print(f"System: {system_prompt}")
        print(f"User: {user_prompt}")
        print("\nNote: Actual API calls require valid API key and will incur costs.")

