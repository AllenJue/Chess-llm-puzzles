#!/usr/bin/env python3
"""
Analyze token usage differences between paradigms for specific models.
"""
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def analyze_model_tokens(model_name: str, model_file_prefix: str):
    """Analyze token usage for a specific model across paradigms."""
    base_dir = parent_dir / "data" / "test_results"
    
    print("="*80)
    print(f"TOKEN ANALYSIS: {model_name}")
    print("="*80)
    
    # Load all three paradigms
    single_file = base_dir / "single_50" / f"test_results_{model_file_prefix}_single_50.csv"
    sc_file = base_dir / "self_consistency_50" / f"test_results_{model_file_prefix}_self_consistency_50.csv"
    debate_file = base_dir / "debate_50" / f"test_results_{model_file_prefix}_debate_50.csv"
    
    results = {}
    
    # Single Model
    if single_file.exists():
        df = pd.read_csv(single_file)
        df = df.head(50)  # First 50 puzzles
        
        total_prompt = df['single_model_prompt_tokens'].fillna(0).sum()
        total_completion = df['single_model_completion_tokens'].fillna(0).sum()
        total_tokens = df['single_model_total_tokens'].fillna(0).sum()
        total_moves = df['total_moves'].sum()
        
        results['single'] = {
            'prompt': total_prompt,
            'completion': total_completion,
            'total': total_tokens,
            'moves': total_moves,
            'prompt_per_move': total_prompt / total_moves if total_moves > 0 else 0,
            'total_per_move': total_tokens / total_moves if total_moves > 0 else 0,
        }
        
        print(f"\nSINGLE MODEL:")
        print(f"  Total Prompt Tokens: {total_prompt:,}")
        print(f"  Total Completion Tokens: {total_completion:,}")
        print(f"  Total Tokens: {total_tokens:,}")
        print(f"  Total Moves: {total_moves:.0f}")
        print(f"  Prompt Tokens/Move: {results['single']['prompt_per_move']:.1f}")
        print(f"  Total Tokens/Move: {results['single']['total_per_move']:.1f}")
        
        # Show sample responses
        print(f"\n  Sample Responses (first 3 puzzles):")
        for idx in range(min(3, len(df))):
            row = df.iloc[idx]
            response = str(row.get('single_model_response', ''))[:150]
            prompt_tokens = row.get('single_model_prompt_tokens', 0)
            completion_tokens = row.get('single_model_completion_tokens', 0)
            print(f"    Puzzle {idx+1} (ID: {row['PuzzleId']}):")
            print(f"      Prompt: {prompt_tokens} tokens, Completion: {completion_tokens} tokens")
            print(f"      Response: {response}...")
    
    # Self-Consistency
    if sc_file.exists():
        df = pd.read_csv(sc_file)
        df = df.head(50)  # First 50 puzzles
        
        # Sum individual agent tokens
        agg_p = df['aggressive_prompt_tokens'].fillna(0).sum()
        pos_p = df['positional_prompt_tokens'].fillna(0).sum()
        neu_p = df['neutral_prompt_tokens'].fillna(0).sum()
        agg_c = df['aggressive_completion_tokens'].fillna(0).sum()
        pos_c = df['positional_completion_tokens'].fillna(0).sum()
        neu_c = df['neutral_completion_tokens'].fillna(0).sum()
        
        total_prompt = agg_p + pos_p + neu_p
        total_completion = agg_c + pos_c + neu_c
        total_tokens = total_prompt + total_completion
        total_moves = df['total_moves'].sum()
        
        results['self_consistency'] = {
            'prompt': total_prompt,
            'completion': total_completion,
            'total': total_tokens,
            'moves': total_moves,
            'prompt_per_move': total_prompt / total_moves if total_moves > 0 else 0,
            'total_per_move': total_tokens / total_moves if total_moves > 0 else 0,
            'agg_prompt': agg_p,
            'pos_prompt': pos_p,
            'neu_prompt': neu_p,
        }
        
        print(f"\nSELF-CONSISTENCY:")
        print(f"  Aggressive Prompt Tokens: {agg_p:,}")
        print(f"  Positional Prompt Tokens: {pos_p:,}")
        print(f"  Neutral Prompt Tokens: {neu_p:,}")
        print(f"  Total Prompt Tokens: {total_prompt:,}")
        print(f"  Total Completion Tokens: {total_completion:,}")
        print(f"  Total Tokens: {total_tokens:,}")
        print(f"  Total Moves: {total_moves:.0f}")
        print(f"  Prompt Tokens/Move: {results['self_consistency']['prompt_per_move']:.1f}")
        print(f"  Total Tokens/Move: {results['self_consistency']['total_per_move']:.1f}")
        
        # Show sample responses
        print(f"\n  Sample Responses (first puzzle):")
        if len(df) > 0:
            row = df.iloc[0]
            agg_response = str(row.get('aggressive_response', ''))[:150]
            pos_response = str(row.get('positional_response', ''))[:150]
            neu_response = str(row.get('neutral_response', ''))[:150]
            print(f"    Puzzle 1 (ID: {row['PuzzleId']}):")
            print(f"      Aggressive: {agg_response}...")
            print(f"      Positional: {pos_response}...")
            print(f"      Neutral: {neu_response}...")
    
    # Debate
    if debate_file.exists():
        df = pd.read_csv(debate_file)
        df = df.head(50)  # First 50 puzzles
        
        total_prompt = df['debate_total_prompt_tokens'].fillna(0).sum()
        total_completion = df['debate_total_completion_tokens'].fillna(0).sum()
        total_tokens = df['debate_total_tokens'].fillna(0).sum()
        total_moves = df['total_moves'].sum()
        
        results['debate'] = {
            'prompt': total_prompt,
            'completion': total_completion,
            'total': total_tokens,
            'moves': total_moves,
            'prompt_per_move': total_prompt / total_moves if total_moves > 0 else 0,
            'total_per_move': total_tokens / total_moves if total_moves > 0 else 0,
        }
        
        print(f"\nDEBATE:")
        print(f"  Total Prompt Tokens: {total_prompt:,}")
        print(f"  Total Completion Tokens: {total_completion:,}")
        print(f"  Total Tokens: {total_tokens:,}")
        print(f"  Total Moves: {total_moves:.0f}")
        print(f"  Prompt Tokens/Move: {results['debate']['prompt_per_move']:.1f}")
        print(f"  Total Tokens/Move: {results['debate']['total_per_move']:.1f}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON (Tokens per Move):")
    print(f"{'='*80}")
    if 'single' in results and 'self_consistency' in results:
        sc_ratio = results['self_consistency']['prompt_per_move'] / results['single']['prompt_per_move'] if results['single']['prompt_per_move'] > 0 else 0
        print(f"  SC Prompt / Single Prompt: {sc_ratio:.2f}x")
    if 'self_consistency' in results and 'debate' in results:
        sc_debate_ratio = results['self_consistency']['prompt_per_move'] / results['debate']['prompt_per_move'] if results['debate']['prompt_per_move'] > 0 else 0
        print(f"  SC Prompt / Debate Prompt: {sc_debate_ratio:.2f}x")
    if 'single' in results and 'debate' in results:
        single_debate_ratio = results['single']['prompt_per_move'] / results['debate']['prompt_per_move'] if results['debate']['prompt_per_move'] > 0 else 0
        print(f"  Single Prompt / Debate Prompt: {single_debate_ratio:.2f}x")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_token_differences.py <model_name> <model_file_prefix>")
        print("\nExamples:")
        print("  python3 analyze_token_differences.py 'gpt-3.5-turbo-instruct' 'gpt-3.5-turbo-instruct'")
        print("  python3 analyze_token_differences.py 'arcee-ai/afm-4.5b' 'arcee-ai_afm-4.5b'")
        sys.exit(1)
    
    model_name = sys.argv[1]
    model_file_prefix = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace('/', '_')
    
    analyze_model_tokens(model_name, model_file_prefix)

