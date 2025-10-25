#!/usr/bin/env python3
"""
Example usage of the Chess Game Engine
Demonstrates different ways to play games programmatically
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append('..')
sys.path.append('../MAD')  # Add MAD directory for utils

from chess_game import ChessGameEngine

def example_single_model_game():
    """Example: Play a game with single model approach"""
    print("🎮 Example: Single Model vs Stockfish")
    print("=" * 50)
    
    # Initialize engine with single model
    engine = ChessGameEngine(
        model_name='gpt-3.5-turbo-instruct',
        stockfish_path='stockfish',
        stockfish_time=1.0,
        use_self_consistency=False
    )
    
    # Play game (model as white)
    game_result = engine.play_game(model_plays_white=True)
    
    # Save results
    json_file = engine.save_game(game_result)
    pgn_file = engine.save_pgn(game_result)
    
    print(f"✅ Game completed!")
    print(f"📊 Result: {game_result['result']}")
    print(f"📈 Total moves: {game_result['total_moves']}")
    print(f"💾 Files saved: {json_file}, {pgn_file}")
    
    return game_result

def example_self_consistency_game():
    """Example: Play a game with self-consistency approach"""
    print("\n🎮 Example: Self-Consistency vs Stockfish")
    print("=" * 50)
    
    # Initialize engine with self-consistency
    engine = ChessGameEngine(
        model_name='gpt-3.5-turbo-instruct',
        stockfish_path='stockfish',
        stockfish_time=1.0,
        use_self_consistency=True
    )
    
    # Play game (model as black)
    game_result = engine.play_game(model_plays_white=False)
    
    # Save results
    json_file = engine.save_game(game_result)
    pgn_file = engine.save_pgn(game_result)
    
    print(f"✅ Game completed!")
    print(f"📊 Result: {game_result['result']}")
    print(f"📈 Total moves: {game_result['total_moves']}")
    print(f"💾 Files saved: {json_file}, {pgn_file}")
    
    return game_result

def example_multiple_games():
    """Example: Play multiple games and compare results"""
    print("\n🎮 Example: Multiple Games Comparison")
    print("=" * 50)
    
    results = []
    
    # Game 1: Single model as white
    print("\n--- Game 1: Single Model (White) ---")
    engine1 = ChessGameEngine(use_self_consistency=False)
    result1 = engine1.play_game(model_plays_white=True)
    results.append(("Single Model (White)", result1))
    
    # Game 2: Self-consistency as white
    print("\n--- Game 2: Self-Consistency (White) ---")
    engine2 = ChessGameEngine(use_self_consistency=True)
    result2 = engine2.play_game(model_plays_white=True)
    results.append(("Self-Consistency (White)", result2))
    
    # Game 3: Single model as black
    print("\n--- Game 3: Single Model (Black) ---")
    engine3 = ChessGameEngine(use_self_consistency=False)
    result3 = engine3.play_game(model_plays_white=False)
    results.append(("Single Model (Black)", result3))
    
    # Summary
    print("\n📊 Games Summary:")
    print("=" * 50)
    for name, result in results:
        print(f"{name:25} | {result['result']:30} | Moves: {result['total_moves']:3}")
    
    return results

def main():
    """Main function to run examples"""
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("🎮 Chess Game Engine - Example Usage")
    print("=" * 60)
    
    # Run examples
    try:
        # Example 1: Single model game
        example_single_model_game()
        
        # Example 2: Self-consistency game
        example_self_consistency_game()
        
        # Example 3: Multiple games (uncomment to run)
        # example_multiple_games()
        
    except KeyboardInterrupt:
        print("\n⏹️  Game interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    print("\n🎉 All examples completed!")

if __name__ == "__main__":
    main()
