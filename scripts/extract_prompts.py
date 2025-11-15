#!/usr/bin/env python3
"""
Extract and display prompts from CSV result files.
"""

import csv
import sys
from pathlib import Path

def extract_prompts(csv_file: str, puzzle_index: int = 0):
    """Extract prompts from a CSV file."""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if len(rows) == 0:
                print(f"⚠️  No data in {csv_file}")
                return
            
            if puzzle_index >= len(rows):
                print(f"⚠️  Puzzle index {puzzle_index} out of range (max: {len(rows)-1})")
                return
            
            row = rows[puzzle_index]
            puzzle_id = row.get('PuzzleId', 'N/A')
            
            print(f'\n{"="*80}')
            print(f'FILE: {Path(csv_file).name}')
            print(f'Puzzle ID: {puzzle_id}')
            print(f'Puzzle Index: {puzzle_index}')
            print(f'{"="*80}')
            
            if 'system_prompt' in row:
                sys_prompt = row['system_prompt']
                print(f'\nSYSTEM PROMPT:')
                print(f'{"-"*80}')
                print(sys_prompt)
            else:
                print('⚠️  system_prompt column not found')
            
            if 'user_prompt' in row:
                user_prompt = row['user_prompt']
                print(f'\nUSER PROMPT (PGN):')
                print(f'{"-"*80}')
                # Show full prompt, but truncate if extremely long
                if len(user_prompt) > 1000:
                    print(user_prompt[:1000])
                    print(f'\n... (truncated, total length: {len(user_prompt)} characters)')
                else:
                    print(user_prompt)
            else:
                print('⚠️  user_prompt column not found')
            print()
            
    except Exception as e:
        print(f'❌ Error reading {csv_file}: {e}')
        print()

if __name__ == "__main__":
    csv_files = [
        'chess_puzzles/data/test_results/test_results_arcee-ai_afm-4.5b_single_50.csv',
        'chess_puzzles/data/test_results/test_results_deepseek-ai_deepseek-v3_single_50.csv',
        'chess_puzzles/data/test_results/test_results_google_gemma-3-12b-it_free_single_50.csv',
        'chess_puzzles/data/test_results/test_results_gpt-3.5-turbo-instruct_single_50.csv',
        'chess_puzzles/data/test_results/test_results_meta-llama_llama-3.1-8b-instruct_single_50.csv',
        'chess_puzzles/data/test_results/test_results_meta-llama_llama-3.3-70b-instruct_single_50.csv',
        'chess_puzzles/data/test_results/test_results_mistralai_mistral-small-24b-instruct-2501_single_50.csv',
        'chess_puzzles/data/test_results/test_results_qwen_qwen3-235b-a22b-instruct-2507_single_50.csv',
    ]
    
    puzzle_index = 0
    if len(sys.argv) > 1:
        puzzle_index = int(sys.argv[1])
    
    for csv_file in csv_files:
        full_path = Path(__file__).parent.parent.parent / csv_file
        if full_path.exists():
            extract_prompts(str(full_path), puzzle_index)
        else:
            print(f'⚠️  File not found: {csv_file}')

