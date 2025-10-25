#!/usr/bin/env python3
"""
Test script for the new Chess Debate V2 system
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the path
sys.path.append('.')

def test_debate_v2():
    """Test the new debate system"""
    try:
        from chess_debate_v2 import ChessDebateV2
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment")
            return False
        
        print("Testing Chess Debate V2 system...")
        
        # Create debate instance
        debate = ChessDebateV2(
            model_name='gpt-3.5-turbo-instruct',
            temperature=0.1,
            openai_api_key=api_key,
            max_rounds=3,
            sleep_time=0.1
        )
        
        # Test with a simple chess position
        test_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5"
        test_board_fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        
        print(f"Testing with position: {test_pgn}")
        print(f"Board FEN: {test_board_fen}")
        
        # Run debate
        final_move, history = debate.run_debate(
            user_prompt=test_pgn,
            board_fen=test_board_fen
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"Final move: {final_move}")
        print(f"Debate success: {history['final_result']['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test that the config file loads correctly"""
    try:
        import json
        config_path = 'MAD/utils/config4chess.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Config file loaded successfully")
        print(f"Config keys: {list(config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Chess Debate V2 System ===\n")
    
    # Test config loading
    print("1. Testing config loading...")
    config_ok = test_config_loading()
    print()
    
    if config_ok:
        # Test debate system
        print("2. Testing debate system...")
        debate_ok = test_debate_v2()
        print()
        
        if debate_ok:
            print("🎉 All tests passed! The new debate system is ready to use.")
            print("\nUsage:")
            print("  python main.py --evaluate --debate-v2 --max-puzzles 5")
        else:
            print("❌ Debate system test failed.")
    else:
        print("❌ Config test failed.")
