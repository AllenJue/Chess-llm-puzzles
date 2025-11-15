#!/usr/bin/env python3
"""
Check which models have suspicious results (low move counts, empty token counts) for first 50 puzzles.
"""
import csv
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def analyze_model_results(model_file: Path, paradigm: str, model_name: str):
    """Analyze results for a specific model file."""
    if not model_file.exists():
        return None
    
    results = {
        'model': model_name,
        'paradigm': paradigm,
        'total_puzzles': 0,
        'puzzles_with_moves': 0,
        'puzzles_with_zero_tokens': 0,
        'total_moves': 0,
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'avg_moves_per_puzzle': 0,
        'puzzles_with_errors': 0,
    }
    
    with open(model_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Only analyze first 50 puzzles (take first 50 rows, assuming they're already sorted)
        unique_puzzles = {}
        for row in rows[:50]:  # Take first 50 rows
            puzzle_id = row.get('PuzzleId', '')
            if puzzle_id:
                # If puzzle already seen, aggregate data
                if puzzle_id not in unique_puzzles:
                    unique_puzzles[puzzle_id] = row
                else:
                    # Aggregate moves if multiple rows per puzzle
                    existing_moves = float(unique_puzzles[puzzle_id].get('total_moves', 0) or 0)
                    new_moves = float(row.get('total_moves', 0) or 0)
                    unique_puzzles[puzzle_id]['total_moves'] = str(max(existing_moves, new_moves))
        
        results['total_puzzles'] = len(unique_puzzles)
        
        for puzzle_id, row in unique_puzzles.items():
            try:
                moves = float(row.get('total_moves', 0) or 0)
                error = str(row.get('error', '')).strip()
                has_error = error and error.lower() != 'nan' and error != ''
                
                if moves > 0:
                    results['puzzles_with_moves'] += 1
                    results['total_moves'] += moves
                
                if has_error:
                    results['puzzles_with_errors'] += 1
                
                # Check token counts based on paradigm
                prompt_tokens = 0
                completion_tokens = 0
                
                if paradigm == 'single':
                    prompt_tokens = int(row.get('single_model_prompt_tokens', 0) or 0)
                    completion_tokens = int(row.get('single_model_completion_tokens', 0) or 0)
                elif paradigm == 'self_consistency':
                    agg_p = int(row.get('aggressive_prompt_tokens', 0) or 0)
                    pos_p = int(row.get('positional_prompt_tokens', 0) or 0)
                    neu_p = int(row.get('neutral_prompt_tokens', 0) or 0)
                    prompt_tokens = agg_p + pos_p + neu_p
                    
                    agg_c = int(row.get('aggressive_completion_tokens', 0) or 0)
                    pos_c = int(row.get('positional_completion_tokens', 0) or 0)
                    neu_c = int(row.get('neutral_completion_tokens', 0) or 0)
                    completion_tokens = agg_c + pos_c + neu_c
                elif paradigm == 'debate':
                    prompt_tokens = int(row.get('debate_total_prompt_tokens', 0) or 0)
                    completion_tokens = int(row.get('debate_total_completion_tokens', 0) or 0)
                
                if prompt_tokens == 0 and completion_tokens == 0:
                    results['puzzles_with_zero_tokens'] += 1
                
                results['total_prompt_tokens'] += prompt_tokens
                results['total_completion_tokens'] += completion_tokens
                
            except Exception as e:
                pass
        
        if results['puzzles_with_moves'] > 0:
            results['avg_moves_per_puzzle'] = results['total_moves'] / results['puzzles_with_moves']
    
    return results

def main():
    base_dir = parent_dir / "data" / "test_results"
    
    # Models to check
    models = [
        ('arcee-ai/afm-4.5b', 'arcee-ai_afm-4.5b'),
        ('deepseek-ai/deepseek-v3', 'deepseek-ai_deepseek-v3'),
        ('gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-instruct'),
        ('meta-llama/llama-3.1-8b-instruct', 'meta-llama_llama-3.1-8b-instruct'),
        ('meta-llama/llama-3.3-70b-instruct', 'meta-llama_llama-3.3-70b-instruct'),
        ('mistralai/mistral-small-24b-instruct-2501', 'mistralai_mistral-small-24b-instruct-2501'),
        ('qwen/qwen3-235b-a22b-instruct-2507', 'qwen_qwen3-235b-a22b-instruct-2507'),
    ]
    
    paradigms = [
        ('single', 'single_50'),
        ('self_consistency', 'self_consistency_50'),
        ('debate', 'debate_50'),
    ]
    
    all_results = []
    
    for model_display, model_file_prefix in models:
        for paradigm_name, paradigm_dir in paradigms:
            model_file = base_dir / paradigm_dir / f"test_results_{model_file_prefix}_{paradigm_name}_50.csv"
            result = analyze_model_results(model_file, paradigm_name, model_display)
            if result:
                all_results.append(result)
    
    # Print suspicious results
    print("="*100)
    print("SUSPICIOUS RESULTS ANALYSIS (First 50 Puzzles)")
    print("="*100)
    print("\nModels with suspicious patterns:")
    print("-"*100)
    print(f"{'Model':<50} {'Paradigm':<20} {'Puzzles':<10} {'Moves':<10} {'Zero Tokens':<15} {'Errors':<10} {'Avg Moves':<12}")
    print("-"*100)
    
    suspicious_models = []
    
    for result in all_results:
        # Flag as suspicious if:
        # 1. >50% puzzles have zero tokens
        # 2. <10 puzzles have moves
        # 3. Average moves per puzzle < 1.0
        zero_token_pct = (result['puzzles_with_zero_tokens'] / result['total_puzzles'] * 100) if result['total_puzzles'] > 0 else 0
        is_suspicious = (
            zero_token_pct > 50 or
            result['puzzles_with_moves'] < 10 or
            result['avg_moves_per_puzzle'] < 1.0
        )
        
        if is_suspicious:
            suspicious_models.append(result)
            print(f"{result['model']:<50} {result['paradigm']:<20} {result['total_puzzles']:<10} "
                  f"{result['puzzles_with_moves']:<10} {result['puzzles_with_zero_tokens']:<15} "
                  f"{result['puzzles_with_errors']:<10} {result['avg_moves_per_puzzle']:<12.2f}")
    
    print("\n" + "="*100)
    print("SUMMARY OF SUSPICIOUS MODELS:")
    print("="*100)
    
    if not suspicious_models:
        print("✅ No suspicious models found!")
    else:
        # Group by model
        by_model = {}
        for result in suspicious_models:
            model = result['model']
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)
        
        for model, results_list in sorted(by_model.items()):
            print(f"\n{model}:")
            for r in results_list:
                zero_pct = (r['puzzles_with_zero_tokens'] / r['total_puzzles'] * 100) if r['total_puzzles'] > 0 else 0
                print(f"  {r['paradigm']:20s}: {r['puzzles_with_moves']:3d} puzzles with moves, "
                      f"{r['puzzles_with_zero_tokens']:3d} ({zero_pct:5.1f}%) with zero tokens, "
                      f"{r['avg_moves_per_puzzle']:.2f} avg moves/puzzle")

if __name__ == "__main__":
    main()

