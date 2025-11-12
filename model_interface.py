"""
Model Interface Module

This module handles interactions with OpenAI's API for chess move prediction.
It provides functions to query different models and extract chess moves from responses.
"""

import os
import time
from typing import Optional, Any, Dict
from openai import OpenAI
from chess_utils import extract_predicted_move, san_to_uci


class ChessModelInterface:
    """
    Interface for interacting with OpenAI models for chess move prediction.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo-instruct",
        *,
        max_completion_tokens: int = 640,
        default_temperature: float = 0.1,
        default_top_p: float = 1.0,
        retry_attempts: int = 1,
    ):
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
        self.max_completion_tokens = max_completion_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.retry_attempts = max(0, retry_attempts)
        normalized_name = (model_name or "").lower()
        self.is_chat_model = "instruct" not in normalized_name and not normalized_name.startswith("text-")
 
    def _call_model_endpoint(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> tuple[Optional[str], Optional[Any], Optional[str]]:
        max_tokens = max_tokens or self.max_completion_tokens
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p

        if self.is_chat_model:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print(f"Model response: {response}")
            text_output = ""
            finish_reason = None
            if getattr(response, "choices", None):
                first_choice = response.choices[0]
                message = getattr(first_choice, "message", None)
                text_output = (message.content if message else "") or ""
                finish_reason = getattr(first_choice, "finish_reason", None)
            text_output = text_output.strip()
            return text_output, response, finish_reason

        combined_prompt = system_prompt + "\n" + user_prompt
        response = self.client.completions.create(
            model=self.model_name,
            prompt=combined_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"Model response: {response}")
        text_output = ""
        finish_reason = None
        if getattr(response, "choices", None):
            first_choice = response.choices[0]
            text_output = (getattr(first_choice, "text", "") or "").strip()
            finish_reason = getattr(first_choice, "finish_reason", None)
        return text_output, response, finish_reason

    def query_model_for_move(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Optional[str]:
        try:
            predicted_move_san, _, _ = self._call_model_endpoint(
                system_prompt,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print("Model output:", repr(predicted_move_san))
            return predicted_move_san
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None

    def query_model_for_move_with_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Call OpenAI model with system and user prompts and return predicted move SAN with token info.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            
        Returns:
            tuple[Optional[str], Optional[dict]]: (predicted_move_san, token_info)
        """
        try:
            predicted_move_san, response, finish_reason = self._call_model_endpoint(
                system_prompt,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print("Model output:", repr(predicted_move_san))
            token_info = self._build_token_info(
                response,
                model_override=self.model_name,
                finish_reason_override=finish_reason,
            ) if response is not None else None
            return predicted_move_san, token_info
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None, None

    def get_move_with_extraction(
        self,
        system_prompt: str,
        user_prompt: str,
        current_turn_number: Optional[int] = None,
        is_white_to_move: bool = True,
        use_gpt4: bool = False,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        retry_attempts: Optional[int] = None,
        force_fallback_move: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[dict]]:
        """
        Get move from model and extract SAN move from response.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            current_turn_number (Optional[int]): Current turn number
            is_white_to_move (bool): Whether it's white's turn
            use_gpt4 (bool): Whether to use GPT-4 instead of instruct model
            
        Returns:
            tuple[Optional[str], Optional[str], Optional[dict]]: (raw_response, extracted_san_move, token_info)
        """
        max_tokens = max_tokens or self.max_completion_tokens
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        retries = self.retry_attempts if retry_attempts is None else max(0, retry_attempts)

        attempt = 0
        response_text: Optional[str] = None
        token_info: Optional[Dict[str, Any]] = None
        extracted_san: Optional[str] = None
        base_prompt = system_prompt
        retry_instruction = (
            "\n\nIMPORTANT: Respond with the exact format:\n"
            "Move: <your move in SAN>\n"
            "Plan: <comma-separated SAN moves you expect afterwards (optional)>\n"
            "Always provide a move, even if you must guess."
        )

        while attempt <= retries:
            active_system_prompt = base_prompt if attempt == 0 else base_prompt + retry_instruction
            response_text, token_info = self.query_model_for_move_with_tokens(
                active_system_prompt,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            if response_text:
                extracted_san = extract_predicted_move(
                    response_text=response_text,
                    current_turn_number=current_turn_number,
                    is_white_to_move=is_white_to_move,
                )
                if extracted_san:
                    return response_text, extracted_san, token_info

            attempt += 1
            temperature = max(0.0, temperature - 0.05)

        fallback_move = force_fallback_move or ("Nf3" if is_white_to_move else "Nf6")
        fallback_response = f"Move: {fallback_move}"
        fallback_token_info = token_info if token_info is not None else self._empty_token_info("fallback")
        return fallback_response, fallback_move, fallback_token_info

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
            system_prompt,
            user_prompt,
            current_turn_number,
            is_white_to_move,
            use_gpt4,
        )
        
        if not san_move:
            return None
        
        return san_to_uci(current_fen, san_move)

    def _empty_token_info(self, finish_reason: str = "fallback") -> Dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": self.model_name,
            "finish_reason": finish_reason,
        }

    def _build_token_info(
        self,
        response: Any,
        *,
        model_override: Optional[str] = None,
        finish_reason_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage else prompt_tokens + completion_tokens
        if total_tokens == 0 and prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens

        finish_reason = finish_reason_override
        if finish_reason is None and getattr(response, "choices", None):
            first_choice = response.choices[0]
            finish_reason = getattr(first_choice, "finish_reason", None)

        return {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
            "model": model_override or self.model_name,
            "finish_reason": finish_reason,
        }


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
    interface = ChessModelInterface(api_key=api_key, model_name="gpt-4-turbo")
    return interface.query_model_for_move(system_prompt, user_prompt)


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

