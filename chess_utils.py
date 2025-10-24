"""
Chess Utilities Module

This module contains utility functions for chess-related operations including
PGN processing, move extraction, and board state management.
"""

import re
import requests
import chess
import chess.pgn
from io import StringIO
from typing import Optional, Tuple


def extract_game_id_and_move(url: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract the game ID (8-chars) and move number (#ply) from any Lichess URL variant:
      - https://lichess.org/4MWQCxQ6#32
      - https://lichess.org/4MWQCxQ6/black#32
      - https://lichess.org/4MWQCxQ6/white#10
    
    Args:
        url (str): Lichess game URL
        
    Returns:
        Tuple[Optional[str], Optional[int]]: (game_id, move_number)
    """
    # Match 8-character game ID
    match = re.search(r"lichess\.org/([A-Za-z0-9]{8})", url)
    game_id = match.group(1) if match else None

    # Extract move number (if present after "#")
    move_match = re.search(r"#([0-9]+)", url)
    move_number = int(move_match.group(1)) if move_match else None

    return game_id, move_number


def get_clean_pgn(game_id: str) -> str:
    """
    Fetch PGN from the Lichess API without evals or clocks.
    
    Args:
        game_id (str): Lichess game ID
        
    Returns:
        str: Clean PGN string
        
    Raises:
        ValueError: If API request fails
    """
    url = f"https://lichess.org/game/export/{game_id}"
    params = {"evals": "false", "clocks": "false"}
    headers = {"Accept": "application/x-chess-pgn"}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        return res.text
    raise ValueError(f"Failed to fetch PGN for {game_id}: {res.status_code}")


def truncate_pgn_at_move(pgn_text: str, move_number: int) -> str:
    """
    Truncate a PGN string at a given ply (half-move) count.
    Example: move_number=32 means 16 moves by each side.
    
    Args:
        pgn_text (str): Full PGN string
        move_number (int): Number of plies to keep
        
    Returns:
        str: Truncated PGN string
    """
    game = chess.pgn.read_game(StringIO(pgn_text))
    board = game.board()

    truncated_game = chess.pgn.Game()
    truncated_game.headers = game.headers.copy()

    node = truncated_game
    move_count = 0
    for move in game.mainline_moves():
        move_count += 1
        if move_count > move_number:
            break
        node = node.add_main_variation(move)
        board.push(move)

    # Return truncated PGN
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return truncated_game.accept(exporter)


def get_partial_pgn_from_url(url: str) -> Tuple[str, Optional[int]]:
    """
    Combine everything: extract ID, download, and truncate PGN.
    
    Args:
        url (str): Lichess game URL
        
    Returns:
        Tuple[str, Optional[int]]: (pgn, move_number)
        
    Raises:
        ValueError: If URL is invalid
    """
    game_id, move_num = extract_game_id_and_move(url)
    if not game_id:
        raise ValueError(f"Invalid Lichess URL: {url}")

    print(f"Fetching PGN for {game_id} up to move #{move_num}")
    pgn = get_clean_pgn(game_id)

    if move_num:
        pgn = truncate_pgn_at_move(pgn, move_num)

    return pgn, move_num


def build_chess_prompts(pgn_partial: str) -> Tuple[str, str]:
    """
    Build system and user prompts for the language model from a PGN partial.

    Args:
        pgn_partial (str): A PGN string (possibly with metadata).

    Returns:
        Tuple[str, str]: (system_prompt, user_prompt)
    """
    system_prompt = """You are a chess grandmaster.
You will be given a partially completed game.
After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.
Use standard algebraic notation, e.g. "e4" or "Rdf8" or "R1a3".
ALWAYS repeat the entire representation of the game so far.
NEVER explain your choice.
"""
    pgn_io = StringIO(pgn_partial)
    game = chess.pgn.read_game(pgn_io)

    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_without_metadata = game.accept(exporter)

    # Remove result suffix from PGN if present
    result_pattern = r'\s*(1-0|0-1|1/2-1/2|\*)\s*$'
    clean_pgn = re.sub(result_pattern, '', pgn_without_metadata)

    user_prompt = clean_pgn.strip()

    return system_prompt, user_prompt


def get_clean_pgn_from_pgn(pgn: str) -> str:
    """
    Clean PGN by removing metadata and result suffixes.
    
    Args:
        pgn (str): Raw PGN string
        
    Returns:
        str: Clean PGN string
    """
    pgn_io = StringIO(pgn)
    game = chess.pgn.read_game(pgn_io)

    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_without_metadata = game.accept(exporter)

    # Remove result suffix from PGN if present
    result_pattern = r'\s*(1-0|0-1|1/2-1/2|\*)\s*$'
    clean_pgn = re.sub(result_pattern, '', pgn_without_metadata)
    return clean_pgn


def san_to_uci(current_fen: str, san_move: str) -> Optional[str]:
    """
    Convert SAN move to UCI move given the current board position in FEN.

    Args:
        current_fen (str): Current board position in FEN notation.
        san_move (str): Move in SAN notation (e.g. 'Qe1#', 'Kxe1', 'e4').

    Returns:
        Optional[str]: UCI move string (e.g. 'e2e4', 'e1e2'), or None if invalid.
    """
    board = chess.Board(current_fen)
    # Remove check/mate symbols from SAN move (like # or +)
    clean_san = san_move.rstrip('#+')
    print(f"Converting SAN '{san_move}' -> '{clean_san}' on board FEN: {current_fen}")
    try:
        move = board.parse_san(clean_san)
        print(f"Parsed move: {move}")
        uci_move = move.uci()
        print(f"UCI move: {uci_move}")
        return uci_move
    except ValueError as e:
        # Invalid SAN, cannot convert
        print(f"ValueError converting SAN '{clean_san}': {e}")
        return None


def extract_predicted_move(response_text: str, current_turn_number: Optional[int] = None, 
                         is_white_to_move: bool = True) -> Optional[str]:
    """
    Robust SAN extractor with preference rules:
      1) explicit labelled predictions like "Predicted move san: Qe1#"
      2) SAN immediately followed by game result (e.g. "Qe1# 0-1")
      3) explicit numbered moves for the provided current_turn_number ("22..." or "22.")
      4) SANs whose nearest preceding move-number <= current_turn_number
      5) ellipsis-based heuristics and safe fallbacks
    
    Args:
        response_text (str): Model response text
        current_turn_number (Optional[int]): Current turn number
        is_white_to_move (bool): Whether it's white's turn to move
        
    Returns:
        Optional[str]: Extracted SAN move or None
    """
    if not response_text or not isinstance(response_text, str):
        return None

    text = response_text.strip()
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    text = re.sub(r'Model output:|Predicted Response:|Completion\(.*?\):?', '', text, flags=re.I)
    # keep result tokens for detection (we'll remove elsewhere as needed)
    text = re.sub(r'\{.*?\}|\(.*?\)', ' ', text)  # remove comments
    text = text.strip()

    # SAN pattern for matching chess moves
    SAN_SUB = r'(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x?[a-h][1-8][+#]?|[a-h][1-8][+#]?)'

    # --- 1) explicit labelled prediction e.g. "Predicted move san: Qe1#" ---
    labelled = re.search(r'predicted\s*move\s*(?:san)?\s*[:\-]\s*(' + SAN_SUB + r')', text, flags=re.I)
    if labelled:
        return labelled.group(1).strip()

    # --- 2) SAN followed by a result token (likely a final predicted move) ---
    san_with_result = re.search(r'(' + SAN_SUB + r')\s+(?:0-1|1-0|1/2-1/2)\b', text, flags=re.I)
    if san_with_result:
        return san_with_result.group(1).strip()

    # Precompute all SAN tokens and their positions
    san_pattern = re.compile(SAN_SUB, flags=re.I)
    san_matches = list(san_pattern.finditer(text))
    san_list = [m.group(0).strip() for m in san_matches]
    san_positions = [m.start() for m in san_matches]

    # Precompute move-number tokens and their positions (e.g. "22.", "22...")
    moveno_pattern = re.compile(r'\b(\d+)\.(?:\.\.)?', flags=re.I)
    moveno_matches = list(moveno_pattern.finditer(text))
    moveno_list = [(int(m.group(1)), m.start(), m.end()) for m in moveno_matches]

    # Helper: find nearest preceding move-number for a given position
    def preceding_moveno(pos):
        prev = None
        for num, s, e in moveno_list:
            if s <= pos:
                prev = (num, s, e)
            else:
                break
        return prev  # (num, start, end) or None

    # --- 3) explicit numbered move for current_turn_number if provided ---
    if current_turn_number is not None:
        turn = int(current_turn_number)
        # for black: look for "22..." pattern
        if not is_white_to_move:
            m = re.search(rf'\b{turn}\.\.\.\s*({SAN_SUB})', text, flags=re.I)
            if m:
                return m.group(1).strip()
        else:
            m = re.search(rf'\b{turn}\.\s*({SAN_SUB})', text, flags=re.I)
            if m:
                return m.group(1).strip()

    # --- 4) prefer SANs whose nearest preceding move-number <= current_turn_number ---
    if current_turn_number is not None and san_matches:
        candidates = []
        for san, pos in zip(san_list, san_positions):
            pm = preceding_moveno(pos)
            if pm is None:
                # treat as early in text -> good candidate
                candidates.append((san, pos, -1))
            else:
                pm_num = pm[0]
                candidates.append((san, pos, pm_num))
        # prefer SANs with pm_num <= current_turn_number, and among them choose the rightmost
        valid = [c for c in candidates if c[2] == -1 or c[2] <= current_turn_number]
        if valid:
            # If black to move, prefer the last valid SAN that is plausible
            # If white to move, prefer the first valid SAN (white move usually appears before reply)
            if is_white_to_move:
                return valid[0][0]
            else:
                return valid[-1][0]

    # --- 5) ellipsis-based explicit black moves anywhere in text ---
    black_pattern = re.compile(r'\b\d+\.\.\.\s*(' + SAN_SUB + r')', flags=re.I)
    black_moves = [m.group(1).strip() for m in black_pattern.finditer(text)]
    if black_moves and not is_white_to_move:
        return black_moves[-1]

    # --- 6) generic explicit white pattern ---
    white_pattern = re.compile(r'\b\d+\.\s*(' + SAN_SUB + r')', flags=re.I)
    white_moves = [m.group(1).strip() for m in white_pattern.finditer(text)]
    if white_moves and is_white_to_move:
        return white_moves[-1]

    # --- 7) Fallback heuristics ---
    if not san_list:
        return None

    # if model is white-to-move pick the first SAN (white tends to appear first)
    if is_white_to_move:
        return san_list[0]

    # if model is black-to-move, prefer penultimate SAN (common pattern "BlackMove WhiteReply ...")
    if len(san_list) >= 2:
        return san_list[-2]

    # last resort: return the last SAN
    return san_list[-1]


def get_board_from_pgn(pgn_text: str) -> chess.Board:
    """
    Get chess board from PGN text.
    
    Args:
        pgn_text (str): PGN string
        
    Returns:
        chess.Board: Chess board after playing all moves
    """
    pgn_io = StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def get_fen_from_pgn(pgn_text: str) -> str:
    """
    Get FEN string from PGN text.
    
    Args:
        pgn_text (str): PGN string
        
    Returns:
        str: FEN string of final position
    """
    board = get_board_from_pgn(pgn_text)
    return board.fen()


if __name__ == "__main__":
    # Example usage and tests
    print("Chess Utils Module - Example Usage")
    print("=" * 40)
    
    # Test SAN extraction
    test_cases = [
        ("22... Qxh2+ 23. Kf1 Qh1#", 22, False, "Qxh2+"),
        ("28... Qe1#", 28, False, "Qe1#"),
        ("15. Rxg7+ Kh8 16. Rxf7+ Kg8", 15, True, "Rxg7+"),
        ("Qe1# 0-1 29. Kxe1", 28, False, "Qe1#")
    ]
    
    for text, turn_num, is_white, expected in test_cases:
        result = extract_predicted_move(text, current_turn_number=turn_num, is_white_to_move=is_white)
        print(f"Input: {text}")
        print(f"Expected: {expected}, Got: {result}")
        print("---")

