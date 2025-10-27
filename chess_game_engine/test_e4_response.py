#!/usr/bin/env python3
"""
Test script to verify that the model responds with e4 for the first move
and investigate the Qg2 repetitive pattern issue
"""

import os
import sys
import chess
import chess.pgn
from io import StringIO

# Add parent directory to path for imports
sys.path.append('..')
sys.path.append('../MAD')  # Add MAD directory for utils

from model_interface import ChessModelInterface
from chess_utils import extract_predicted_move, san_to_uci

def test_e4_response():
    """Test that the model responds with e4 for the first move"""
    print("🧪 Testing e4 Response for First Move")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to run this test")
        return False
    
    try:
        # Create model interface
        model_interface = ChessModelInterface(
            model_name='gpt-3.5-turbo-instruct',
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create initial board position
        board = chess.Board()
        
        # System prompt (same as chess_game.py)
        system_prompt = (
            "You are a chess grandmaster.\n"
            "You will be given a partially completed game.\n"
            "After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.\n"
            "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'."
        )
        
        # User prompt - empty game (first move)
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        current_game = chess.pgn.Game.from_board(board)
        user_prompt = current_game.accept(exporter)
        
        print(f"System prompt: {repr(system_prompt)}")
        print(f"User prompt: {repr(user_prompt)}")
        print()
        
        # Get model response
        print("🤖 Querying model for first move...")
        raw_response, predicted_san, token_info = model_interface.get_move_with_extraction(
            system_prompt, user_prompt,
            current_turn_number=1,
            is_white_to_move=True
        )
        
        print(f"Raw response: {repr(raw_response)}")
        print(f"Extracted SAN: {repr(predicted_san)}")
        print(f"Token info: {token_info}")
        print()
        
        # Test the move extraction with the long repetitive sequence
        print("🧪 Testing move extraction with long repetitive sequence...")
        long_sequence = "'1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6\n5. O-O Be7\n6. Re1 b5\n7. Bb3 O-O\n8. c3 d5\n9. exd5 Nxd5\n10. Nxe5 Nxe5\n11. Rxe5 c6\n12. d4 Bd6\n13. Re1 Qh4\n14. g3 Qh3\n15. Be3 Bg4\n16. Qd3 Rae8\n17. Nd2 Re6\n18. Qf1 Qh5\n19. a4 Rfe8\n20. axb5 axb5\n21. Bxd5 Qxd5\n22. Qg2 Qh5\n23. Ra6 Bf8\n24. Rxc6 Bh3\n25. Qf3 Bg4\n26. Qg2 Bh3\n27. Qf3 Bg4\n28. Qg2 Bh3\n29. Qf3 Bg4\n30. Qg2 Bh3\n31. Qf3 Bg4\n32. Qg2 Bh3\n33. Qf3 Bg4\n34. Qg2 Bh3\n35. Qf3 Bg4\n36. Qg2 Bh3\n37. Qf3 Bg4\n38. Qg2 Bh3\n39. Qf3 Bg4\n40. Qg2 Bh3\n41. Qf3 Bg4\n42. Qg2 Bh3\n43. Qf3 Bg4\n44. Qg2 Bh3\n45. Qf3 Bg4\n46. Qg2 Bh3\n47. Qf3 Bg4\n48. Qg2 Bh3\n49. Qf3 Bg4\n50. Qg2 Bh3\n51. Qf3 Bg4\n52. Qg2 Bh3\n53. Qf3 Bg4\n54. Qg2 Bh3\n55. Qf3 Bg4\n56. Qg2 Bh3\n57. Qf3 Bg4\n58. Qg2 Bh3\n59. Qf3 Bg4\n60. Qg2 Bh3'"
        
        extracted_e4 = extract_predicted_move(long_sequence, current_turn_number=1, is_white_to_move=True)
        print(f"Extracted move from long sequence (turn 1, white): {repr(extracted_e4)}")
        
        # Test various moves from the sequence
        test_cases = [
            (1, True, "e4"),
            (2, True, "Nf3"), 
            (3, True, "Bb5"),
            (22, True, "Qg2"),
            (23, False, "Bf8"),
            (25, True, "Qf3"),
            (26, False, "Bg4"),
            (60, True, "Qg2"),
        ]
        
        print("\n🧪 Testing various moves from the sequence:")
        for turn, is_white, expected in test_cases:
            extracted = extract_predicted_move(long_sequence, current_turn_number=turn, is_white_to_move=is_white)
            status = "✅" if extracted == expected else "❌"
            print(f"  {status} Turn {turn} ({'White' if is_white else 'Black'}): Expected {expected}, Got {extracted}")
        
        # Check if the model actually responds with e4
        success = predicted_san == "e4"
        if success:
            print(f"\n✅ SUCCESS: Model correctly responded with 'e4' for the first move")
        else:
            print(f"\n❌ FAILURE: Model responded with '{predicted_san}' instead of 'e4'")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qg2_pattern():
    """Test the specific Qg2/Qf3 repetitive pattern"""
    print("\n🧪 Testing Qg2/Qf3 Repetitive Pattern")
    print("=" * 50)
    
    # Test the repetitive pattern from the terminal output
    repetitive_text = "Qg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3\nQf3 Bg4\nQg2 Bh3"
    
    print("Testing move extraction from repetitive pattern:")
    test_cases = [
        (60, True, "Qg2"),   # White to move
        (61, False, "Qf3"),  # Black to move
        (62, True, "Qg2"),   # White to move
        (63, False, "Qf3"),  # Black to move
    ]
    
    for turn, is_white, expected in test_cases:
        extracted = extract_predicted_move(repetitive_text, current_turn_number=turn, is_white_to_move=is_white)
        status = "✅" if extracted == expected else "❌"
        print(f"  {status} Turn {turn} ({'White' if is_white else 'Black'}): Expected {expected}, Got {extracted}")

if __name__ == "__main__":
    print("🎯 Chess Move Extraction Test Suite")
    print("=" * 60)
    
    # Test e4 response
    e4_success = test_e4_response()
    
    # Test Qg2 pattern
    test_qg2_pattern()
    
    print(f"\n🎯 Overall Result: {'✅ PASSED' if e4_success else '❌ FAILED'}")
