#!/usr/bin/env python3
"""
Analyze and display errors for each model from single model results.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def extract_model_name(filename: str) -> str:
    """Extract a clean model name from filename."""
    name = filename.replace("test_results_", "").replace("_single_50.csv", "")
    # Clean up model name for display
    name = name.replace("_", "/").replace(":free", " (free)")
    return name

def analyze_errors(csv_file: Path, num_puzzles: int = 50) -> Dict:
    """Analyze errors from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None
        
        # Get unique puzzles in order of first appearance
        unique_puzzles = df['PuzzleId'].drop_duplicates().head(num_puzzles).tolist()
        actual_num_puzzles = len(unique_puzzles)
        
        if actual_num_puzzles == 0:
            return None
        
        # Collect errors
        errors = []
        error_types = []
        puzzles_with_errors = 0
        
        for puzzle_id in unique_puzzles:
            puzzle_rows = df[df['PuzzleId'] == puzzle_id]
            # Get the first row for this puzzle (should have the error if any)
            first_row = puzzle_rows.iloc[0]
            
            error = first_row.get('error', '')
            if pd.notna(error) and error and error.strip():
                errors.append({
                    'puzzle_id': puzzle_id,
                    'error': str(error).strip()
                })
                puzzles_with_errors += 1
                
                # Categorize error type
                error_str = str(error).lower()
                if 'failed to extract' in error_str:
                    error_types.append('Failed to extract move')
                elif 'mismatch' in error_str:
                    error_types.append('Move mismatch')
                elif 'illegal' in error_str or 'invalid' in error_str:
                    error_types.append('Illegal/invalid move')
                elif 'api' in error_str or 'error calling' in error_str:
                    error_types.append('API error')
                elif 'timeout' in error_str:
                    error_types.append('Timeout')
                else:
                    error_types.append('Other')
        
        error_counts = Counter(error_types)
        
        return {
            'total_puzzles': actual_num_puzzles,
            'puzzles_with_errors': puzzles_with_errors,
            'puzzles_without_errors': actual_num_puzzles - puzzles_with_errors,
            'error_rate': (puzzles_with_errors / actual_num_puzzles * 100) if actual_num_puzzles > 0 else 0,
            'errors': errors,
            'error_type_counts': dict(error_counts),
            'total_errors': len(errors)
        }
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_errors(results_dir: str = "data/test_results"):
    """Display errors for all single model results."""
    
    results_path = parent_dir / results_dir
    
    # Find all single model CSV files
    csv_files = list(results_path.glob("test_results_*_single_50.csv"))
    
    if len(csv_files) == 0:
        print(f"No single model result files found in {results_path}")
        return
    
    print(f"Found {len(csv_files)} single model result files\n")
    print("="*80)
    print("ERROR ANALYSIS (First 50 Puzzles)")
    print("="*80)
    print()
    
    all_results = {}
    
    # Analyze each model
    for csv_file in csv_files:
        model_name = extract_model_name(csv_file.name)
        error_data = analyze_errors(csv_file, num_puzzles=50)
        if error_data:
            all_results[model_name] = error_data
    
    # Sort by error rate (highest first)
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['error_rate'], reverse=True)
    
    # Summary table
    print("ERROR SUMMARY BY MODEL")
    print("-"*80)
    print(f"{'Model':<50} {'Errors':<10} {'Error Rate':<15}")
    print("-"*80)
    for model_name, data in sorted_models:
        error_info = f"{data['puzzles_with_errors']}/{data['total_puzzles']}"
        error_rate = f"{data['error_rate']:.1f}%"
        print(f"{model_name:<50} {error_info:<10} {error_rate:<15}")
    print()
    
    # Detailed error breakdown by type
    print("="*80)
    print("ERROR TYPE BREAKDOWN")
    print("="*80)
    print()
    
    for model_name, data in sorted_models:
        if data['total_errors'] > 0:
            print(f"\n{model_name}")
            print("-" * 80)
            print(f"Total errors: {data['total_errors']} out of {data['total_puzzles']} puzzles ({data['error_rate']:.1f}%)")
            print()
            print("Error types:")
            for error_type, count in sorted(data['error_type_counts'].items(), key=lambda x: x[1], reverse=True):
                pct = (count / data['total_errors'] * 100) if data['total_errors'] > 0 else 0
                print(f"  {error_type:<30} {count:3d} ({pct:5.1f}%)")
    
    # Show sample errors for each model
    print("\n" + "="*80)
    print("SAMPLE ERRORS (First 5 per model)")
    print("="*80)
    print()
    
    for model_name, data in sorted_models:
        if data['total_errors'] > 0:
            print(f"\n{model_name} - Sample Errors:")
            print("-" * 80)
            sample_errors = data['errors'][:5]  # First 5 errors
            for i, err in enumerate(sample_errors, 1):
                error_text = err['error']
                # Truncate long errors
                if len(error_text) > 100:
                    error_text = error_text[:97] + "..."
                print(f"  {i}. Puzzle {err['puzzle_id']}: {error_text}")
            
            if len(data['errors']) > 5:
                print(f"  ... and {len(data['errors']) - 5} more errors")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze errors from single model results")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                       help="Directory containing result CSV files")
    
    args = parser.parse_args()
    display_errors(args.results_dir)

