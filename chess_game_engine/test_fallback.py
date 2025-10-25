#!/usr/bin/env python3
"""
Test script to demonstrate the fallback system
This script simulates a scenario where the model fails to generate valid moves
"""

import os
import sys
import random

# Add parent directory to path for imports
sys.path.append('..')
sys.path.append('../MAD')  # Add MAD directory for utils

from chess_game import ChessGameEngine

def test_fallback_system():
    """Test the fallback system by simulating model failures"""
    print("🧪 Testing Fallback System")
    print("=" * 50)
    
    # Create a game engine
    engine = ChessGameEngine(
        model_name='gpt-3.5-turbo-instruct',
        stockfish_path='stockfish',
        stockfish_time=0.1,  # Very fast for testing
        use_self_consistency=False
    )
    
    # Mock the model interface to simulate failures
    original_get_move = engine.model_interface.get_move_with_extraction
    
    def mock_get_move_with_failures(system_prompt, user_prompt, **kwargs):
        """Mock function that fails 50% of the time"""
        if random.random() < 0.5:  # 50% chance of failure
            print("🎲 Simulating model failure...")
            return None, None, None  # Simulate failure
        else:
            return original_get_move(system_prompt, user_prompt, **kwargs)
    
    # Replace the method temporarily
    engine.model_interface.get_move_with_extraction = mock_get_move_with_failures
    
    print("🎮 Starting test game with simulated failures...")
    print("🤖 Model will fail ~50% of the time to test fallback system")
    print("🐟 Stockfish will play very fast (0.1s per move)")
    print()
    
    # Play a short game
    try:
        result = engine.play_game(model_plays_white=True)
        
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS")
        print("=" * 50)
        print(f"Game result: {result['result']}")
        print(f"Total moves: {result['total_moves']}")
        print(f"Fallback moves used: {result['fallback_count']}")
        
        if result['fallback_moves']:
            print("\n🎲 Fallback moves details:")
            for i, fallback in enumerate(result['fallback_moves'], 1):
                print(f"  {i}. Move {fallback['move_number']}: {fallback['reason']}")
                print(f"     Fallback move: {fallback['fallback_move']}")
                if fallback['model_response']:
                    print(f"     Model response: {fallback['model_response'][:100]}...")
                print()
        
        # Show move history with fallback indicators
        print("📝 Move history:")
        for move in result['game_history']:
            if move['is_fallback_move']:
                print(f"  Move {move['move_number']}: {move['move_san']} (FALLBACK)")
            else:
                print(f"  Move {move['move_number']}: {move['move_san']}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"🎯 Fallback system working: {result['fallback_count']} fallback moves used")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to run this test")
        sys.exit(1)
    
    test_fallback_system()
