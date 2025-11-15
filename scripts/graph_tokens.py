#!/usr/bin/env python3
"""
Graph token usage (prompt and completion) for all single model results.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def extract_model_name(filename: str) -> str:
    """Extract a clean model name from filename."""
    name = filename.replace("test_results_", "").replace("_single_50.csv", "")
    # Clean up model name for display
    name = name.replace("_", "/").replace(":free", " (free)")
    return name

def load_and_calculate_tokens(csv_file: Path, num_puzzles: int = 50) -> Dict:
    """Load a CSV and calculate total token usage."""
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None
        
        # Get unique puzzles in order of first appearance
        unique_puzzles = df['PuzzleId'].drop_duplicates().head(num_puzzles).tolist()
        actual_num_puzzles = len(unique_puzzles)
        
        if actual_num_puzzles == 0:
            return None
        
        # Sum up tokens across all rows for the first N puzzles
        puzzle_rows = df[df['PuzzleId'].isin(unique_puzzles)]
        
        # Calculate total tokens
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        # Try different column names for token counts
        prompt_cols = ['single_model_prompt_tokens', 'prompt_tokens', 'total_prompt_tokens']
        completion_cols = ['single_model_completion_tokens', 'completion_tokens', 'total_completion_tokens']
        total_cols = ['single_model_total_tokens', 'total_tokens']
        
        prompt_col = None
        completion_col = None
        total_col = None
        
        for col in prompt_cols:
            if col in puzzle_rows.columns:
                prompt_col = col
                break
        
        for col in completion_cols:
            if col in puzzle_rows.columns:
                completion_col = col
                break
        
        for col in total_cols:
            if col in puzzle_rows.columns:
                total_col = col
                break
        
        if prompt_col:
            # Sum all prompt tokens (convert to numeric, handling any non-numeric values)
            total_prompt_tokens = pd.to_numeric(puzzle_rows[prompt_col], errors='coerce').fillna(0).sum()
        
        if completion_col:
            # Sum all completion tokens
            total_completion_tokens = pd.to_numeric(puzzle_rows[completion_col], errors='coerce').fillna(0).sum()
        
        if total_col:
            # Sum all total tokens
            total_tokens = pd.to_numeric(puzzle_rows[total_col], errors='coerce').fillna(0).sum()
        elif total_prompt_tokens > 0 and total_completion_tokens > 0:
            # Calculate total if not directly available
            total_tokens = total_prompt_tokens + total_completion_tokens
        
        return {
            'total_prompt_tokens': int(total_prompt_tokens),
            'total_completion_tokens': int(total_completion_tokens),
            'total_tokens': int(total_tokens),
            'total_puzzles': actual_num_puzzles,
            'avg_prompt_tokens_per_puzzle': int(total_prompt_tokens / actual_num_puzzles) if actual_num_puzzles > 0 else 0,
            'avg_completion_tokens_per_puzzle': int(total_completion_tokens / actual_num_puzzles) if actual_num_puzzles > 0 else 0,
        }
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_token_graphs(results_dir: str = "data/test_results", output_dir: str = "data/graphs"):
    """Create token usage graphs for all single model results."""
    
    results_path = parent_dir / results_dir
    output_path = parent_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all single model CSV files (check both root and single_50 subdirectory)
    csv_files = list(results_path.glob("test_results_*_single_50.csv"))
    single_50_dir = results_path / "single_50"
    if single_50_dir.exists():
        csv_files.extend(list(single_50_dir.glob("test_results_*_single_50.csv")))
    
    if len(csv_files) == 0:
        print(f"No single model result files found in {results_path}")
        return
    
    print(f"Found {len(csv_files)} single model result files")
    
    # Models to exclude (failed models with 0 tokens)
    excluded_models = [
        "google/gemma-3-4b-it (free)",
        "google/gemma-3-12b-it (free)",
        "google/gemma-3-4b-it:free",
        "google/gemma-3-12b-it:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "meta-llama/llama-3.3-8b-instruct (free)",
    ]
    
    # Load and calculate tokens for each model
    model_results = {}
    for csv_file in csv_files:
        model_name = extract_model_name(csv_file.name)
        # Skip excluded models
        if model_name in excluded_models:
            print(f"Skipping excluded model: {model_name}")
            continue
        token_data = load_and_calculate_tokens(csv_file, num_puzzles=50)
        if token_data and token_data['total_tokens'] > 0:  # Only include models with actual token usage
            model_results[model_name] = token_data
            print(f"{model_name}: {token_data['total_prompt_tokens']:,} prompt + {token_data['total_completion_tokens']:,} completion = {token_data['total_tokens']:,} total tokens")
    
    if len(model_results) == 0:
        print("No valid results found")
        return
    
    # Sort models by total tokens (descending)
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['total_tokens'], reverse=True)
    
    # Prepare data for plotting
    model_names = [name for name, _ in sorted_models]
    prompt_tokens = [data['total_prompt_tokens'] for _, data in sorted_models]
    completion_tokens = [data['total_completion_tokens'] for _, data in sorted_models]
    total_tokens = [data['total_tokens'] for _, data in sorted_models]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, max(8, len(model_names) * 0.4)))
    
    # Create stacked bars: prompt tokens first (bottom), then completion tokens on top
    # For horizontal bars, "left" parameter positions the second bar after the first
    bars1 = ax.barh(range(len(model_names)), prompt_tokens, label='Prompt Tokens', 
                    color=sns.color_palette("Blues", 3)[1])
    bars2 = ax.barh(range(len(model_names)), completion_tokens, left=prompt_tokens, 
                    label='Completion Tokens', color=sns.color_palette("Oranges", 3)[1])
    
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Total Tokens', fontsize=12, fontweight='bold')
    ax.set_title('Token Usage: Prompt + Completion Tokens by Model (First 50 Puzzles)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (prompt, completion, total) in enumerate(zip(prompt_tokens, completion_tokens, total_tokens)):
        if total > 0:
            # Label on the right side of the bar showing total
            ax.text(total + max(total_tokens) * 0.01, i, f'{total:,}', 
                    va='center', fontsize=9, fontweight='bold')
            # Label in the middle for prompt tokens (bottom stack, if bar is large enough)
            if prompt > max(total_tokens) * 0.1:
                ax.text(prompt / 2, i, f'{prompt:,}', 
                        va='center', ha='center', fontsize=8, color='white', fontweight='bold')
            # Label for completion tokens (top stack, if bar is large enough)
            if completion > max(total_tokens) * 0.1:
                ax.text(prompt + completion / 2, i, f'{completion:,}', 
                        va='center', ha='center', fontsize=8, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = output_path / "single_model_token_usage.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Token graph saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("TOKEN USAGE SUMMARY (First 50 Puzzles)")
    print("="*80)
    print(f"{'Model':<50} {'Prompt Tokens':<20} {'Completion Tokens':<20} {'Total Tokens':<20}")
    print("-"*80)
    for name, data in sorted_models:
        prompt_str = f"{data['total_prompt_tokens']:,}"
        completion_str = f"{data['total_completion_tokens']:,}"
        total_str = f"{data['total_tokens']:,}"
        print(f"{name:<50} {prompt_str:<20} {completion_str:<20} {total_str:<20}")
    
    print("\n" + "="*80)
    print("AVERAGE TOKENS PER PUZZLE")
    print("="*80)
    print(f"{'Model':<50} {'Avg Prompt':<20} {'Avg Completion':<20} {'Avg Total':<20}")
    print("-"*80)
    for name, data in sorted_models:
        avg_prompt = f"{data['avg_prompt_tokens_per_puzzle']:,}"
        avg_completion = f"{data['avg_completion_tokens_per_puzzle']:,}"
        avg_total = f"{(data['avg_prompt_tokens_per_puzzle'] + data['avg_completion_tokens_per_puzzle']):,}"
        print(f"{name:<50} {avg_prompt:<20} {avg_completion:<20} {avg_total:<20}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Graph token usage for single model results")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                       help="Directory containing result CSV files")
    parser.add_argument("--output-dir", type=str, default="data/graphs",
                       help="Directory to save output graphs")
    
    args = parser.parse_args()
    create_token_graphs(args.results_dir, args.output_dir)

