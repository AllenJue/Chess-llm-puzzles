#!/usr/bin/env python3
"""
Script to visualize and validate chess puzzle evaluation results.
Checks data correctness and creates visualizations for accuracy and token usage.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_csv_files(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV result files from the data directory."""
    csv_files = {}
    data_path = Path(data_dir)
    
    for csv_file in data_path.glob("*.csv"):
        # Skip the input puzzle file
        if "lichess_puzzles" in csv_file.name:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            # Extract model name from filename
            model_name = csv_file.stem.replace("test_results_", "").replace("results_", "")
            csv_files[model_name] = df
            print(f"Loaded {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    return csv_files


def load_game_results(games_dir: str) -> List[Dict[str, Any]]:
    """Load JSON game result files for full-game analysis."""
    results: List[Dict[str, Any]] = []
    games_path = Path(games_dir)
    if not games_path.exists():
        return results

    for game_file in games_path.glob("*.json"):
        try:
            with open(game_file, "r") as f:
                data = json.load(f)
            results.append({
                "path": str(game_file),
                "data": data,
            })
            print(f"Loaded game file: {game_file.name}")
        except Exception as e:
            print(f"Error loading game file {game_file}: {e}")
    return results


def analyze_game_result(game_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract token statistics and validation info for a single game result."""
    data = game_entry.get("data", {})
    path = game_entry.get("path", "")

    model_name = data.get("model_name") or data.get("metadata", {}).get("model_name", "unknown")
    approach = data.get("model_approach") or data.get("metadata", {}).get("model_approach", "single_model")
    result = data.get("result", "unknown")

    single_log = data.get("single_model_token_log", []) or []
    sc_log = data.get("self_consistency_token_log", []) or []

    single_prompt_calc = sum(entry.get("prompt_tokens", 0) for entry in single_log)
    single_completion_calc = sum(entry.get("completion_tokens", 0) for entry in single_log)
    single_total_calc = sum(entry.get("total_tokens", entry.get("prompt_tokens", 0) + entry.get("completion_tokens", 0)) for entry in single_log)

    single_prompt_reported = data.get("single_model_total_prompt_tokens", single_prompt_calc)
    single_completion_reported = data.get("single_model_total_completion_tokens", single_completion_calc)
    single_total_reported = data.get("single_model_total_tokens", single_total_calc)

    sc_prompt_calc = sum(entry.get("total_prompt_tokens", 0) for entry in sc_log)
    sc_completion_calc = sum(entry.get("total_completion_tokens", 0) for entry in sc_log)
    sc_total_calc = sum(entry.get("total_tokens", entry.get("total_prompt_tokens", 0) + entry.get("total_completion_tokens", 0)) for entry in sc_log)

    sc_prompt_reported = data.get("self_consistency_total_prompt_tokens", sc_prompt_calc)
    sc_completion_reported = data.get("self_consistency_total_completion_tokens", sc_completion_calc)
    sc_total_reported = data.get("self_consistency_total_tokens", sc_total_calc)

    issues: List[str] = []
    if single_log:
        if single_prompt_calc != single_prompt_reported:
            issues.append(f"Single model prompt tokens mismatch: calc {single_prompt_calc} != reported {single_prompt_reported}")
        if single_completion_calc != single_completion_reported:
            issues.append(f"Single model completion tokens mismatch: calc {single_completion_calc} != reported {single_completion_reported}")
        if single_total_calc != single_total_reported:
            issues.append(f"Single model total tokens mismatch: calc {single_total_calc} != reported {single_total_reported}")

    if sc_log:
        if sc_prompt_calc != sc_prompt_reported:
            issues.append(f"Self-consistency prompt tokens mismatch: calc {sc_prompt_calc} != reported {sc_prompt_reported}")
        if sc_completion_calc != sc_completion_reported:
            issues.append(f"Self-consistency completion tokens mismatch: calc {sc_completion_calc} != reported {sc_completion_reported}")
        if sc_total_calc != sc_total_reported:
            issues.append(f"Self-consistency total tokens mismatch: calc {sc_total_calc} != reported {sc_total_reported}")

    return {
        "file": path,
        "model": model_name,
        "approach": approach,
        "result": result,
        "single_model_prompt_tokens_calc": single_prompt_calc,
        "single_model_completion_tokens_calc": single_completion_calc,
        "single_model_total_tokens_calc": single_total_calc,
        "single_model_prompt_tokens_reported": single_prompt_reported,
        "single_model_completion_tokens_reported": single_completion_reported,
        "single_model_total_tokens_reported": single_total_reported,
        "self_consistency_prompt_tokens_calc": sc_prompt_calc,
        "self_consistency_completion_tokens_calc": sc_completion_calc,
        "self_consistency_total_tokens_calc": sc_total_calc,
        "self_consistency_prompt_tokens_reported": sc_prompt_reported,
        "self_consistency_completion_tokens_reported": sc_completion_reported,
        "self_consistency_total_tokens_reported": sc_total_reported,
        "issues": issues,
    }


def print_game_summary(game_stats: List[Dict[str, Any]]) -> None:
    """Print summary information for analyzed game results."""
    print("\n" + "="*80)
    print("FULL GAME TOKEN SUMMARY")
    print("="*80)

    if not game_stats:
        print("No full game JSON files found.")
        return

    for stats in game_stats:
        print(f"\nFile: {stats['file']}")
        print(f"  Model: {stats['model']}")
        print(f"  Approach: {stats['approach']}")
        print(f"  Result: {stats['result']}")
        if stats["single_model_total_tokens_reported"]:
            print(f"  Single-model tokens (reported): prompt {stats['single_model_prompt_tokens_reported']}, completion {stats['single_model_completion_tokens_reported']}, total {stats['single_model_total_tokens_reported']}")
        if stats["single_model_total_tokens_calc"]:
            print(f"  Single-model tokens (calculated): prompt {stats['single_model_prompt_tokens_calc']}, completion {stats['single_model_completion_tokens_calc']}, total {stats['single_model_total_tokens_calc']}")
        if stats["self_consistency_total_tokens_reported"]:
            print(f"  Self-consistency tokens (reported): prompt {stats['self_consistency_prompt_tokens_reported']}, completion {stats['self_consistency_completion_tokens_reported']}, total {stats['self_consistency_total_tokens_reported']}")
        if stats["self_consistency_total_tokens_calc"]:
            print(f"  Self-consistency tokens (calculated): prompt {stats['self_consistency_prompt_tokens_calc']}, completion {stats['self_consistency_completion_tokens_calc']}, total {stats['self_consistency_total_tokens_calc']}")
        if stats["issues"]:
            print("  ⚠️  Issues:")
            for issue in stats["issues"]:
                print(f"     - {issue}")
        else:
            print("  ✅ Token totals match log calculations")


def validate_data(df: pd.DataFrame, model_name: str) -> Dict[str, any]:
    """Validate data correctness and return validation results."""
    issues = []
    stats = {
        "model": model_name,
        "total_puzzles": len(df),
        "valid_rows": 0,
        "issues": []
    }
    
    paradigm = detect_paradigm(df)
    stats["paradigm"] = paradigm
    
    # Check required columns
    required_cols = ["puzzle_solved", "correct_moves"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        stats["issues"].append(f"Missing required columns: {missing_cols}")
        return stats

    paradigm_required_cols = []
    if paradigm == "single_model":
        paradigm_required_cols = ["single_model_total_tokens"]
    elif paradigm == "self_consistency":
        paradigm_required_cols = ["self_consistency_total_tokens"]
    elif paradigm in ("debate", "debate_v2"):
        paradigm_required_cols = ["debate_total_tokens"]

    missing_paradigm_cols = [col for col in paradigm_required_cols if col not in df.columns]
    if missing_paradigm_cols:
        stats["issues"].append(f"Missing paradigm-specific columns: {missing_paradigm_cols}")
    
    # Check for NaN values in critical columns
    critical_cols = ["puzzle_solved", "correct_moves"]
    for col in critical_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            stats["issues"].append(f"Column '{col}' has {nan_count} NaN values")
    
    # Validate puzzle_solved is boolean
    if df["puzzle_solved"].dtype != bool:
        # Try to convert
        try:
            df["puzzle_solved"] = df["puzzle_solved"].astype(bool)
        except:
            stats["issues"].append("Column 'puzzle_solved' is not boolean")
    
    # Check token columns (more flexible search)
    # First check for per-puzzle token columns (single_model_prompt_tokens, etc.)
    per_puzzle_prompt_cols = [col for col in df.columns if "prompt_tokens" in col.lower() and "total" not in col.lower()]
    per_puzzle_completion_cols = [col for col in df.columns if "completion_tokens" in col.lower() and "total" not in col.lower()]
    
    # Then check for total columns
    prompt_cols = [col for col in df.columns if "prompt_tokens" in col.lower() and "total" in col.lower()]
    completion_cols = [col for col in df.columns if "completion_tokens" in col.lower() and "total" in col.lower()]
    
    if not prompt_cols and not per_puzzle_prompt_cols:
        # Try alternative patterns
        prompt_cols = [col for col in df.columns if "prompt" in col.lower() and "token" in col.lower()]
    if not completion_cols and not per_puzzle_completion_cols:
        completion_cols = [col for col in df.columns if "completion" in col.lower() and "token" in col.lower()]
    
    all_prompt_cols = prompt_cols + per_puzzle_prompt_cols
    all_completion_cols = completion_cols + per_puzzle_completion_cols
    
    if not all_prompt_cols:
        stats["issues"].append("No prompt token columns found")
    else:
        # Check for negative or unreasonably large values
        for col in all_prompt_cols:
            if df[col].dtype in [np.int64, np.float64]:
                negative = (df[col] < 0).sum()
                very_large = (df[col] > 1000000).sum()  # More than 1M tokens seems unreasonable
                if negative > 0:
                    stats["issues"].append(f"Column '{col}' has {negative} negative values")
                if very_large > 0:
                    stats["issues"].append(f"Column '{col}' has {very_large} values > 1M tokens")
    
    if not all_completion_cols:
        stats["issues"].append("No completion token columns found")
    else:
        # Check for negative or unreasonably large values
        for col in all_completion_cols:
            if df[col].dtype in [np.int64, np.float64]:
                negative = (df[col] < 0).sum()
                very_large = (df[col] > 1000000).sum()
                if negative > 0:
                    stats["issues"].append(f"Column '{col}' has {negative} negative values")
                if very_large > 0:
                    stats["issues"].append(f"Column '{col}' has {very_large} values > 1M tokens")
    
    # Validate that solved puzzles have correct_moves > 0
    if "puzzle_solved" in df.columns and "correct_moves" in df.columns:
        # Convert puzzle_solved to boolean if needed
        puzzle_solved_bool = df["puzzle_solved"]
        if puzzle_solved_bool.dtype != bool:
            try:
                puzzle_solved_bool = puzzle_solved_bool.astype(bool)
            except:
                pass
        
        invalid = df[puzzle_solved_bool & (df["correct_moves"] <= 0)]
        if len(invalid) > 0:
            stats["issues"].append(f"{len(invalid)} puzzles marked as solved but have correct_moves <= 0")
    
    # Check token consistency: total should equal prompt + completion (if both exist)
    # Check per-puzzle columns first (single_model_prompt_tokens + single_model_completion_tokens = single_model_total_tokens)
    if per_puzzle_prompt_cols and per_puzzle_completion_cols:
        # Find matching per-puzzle total column (e.g., single_model_total_tokens)
        per_puzzle_total_cols = [col for col in df.columns 
                                if "total_tokens" in col.lower() 
                                and not any(x in col.lower() for x in ["prompt", "completion"])  # Exclude prompt/completion specific columns
                                and any(prefix in col.lower() for prefix in ["single_model", "aggressive", "positional", "neutral", "moderator", "judge"])]
        
        # Match per-puzzle columns by prefix
        for prompt_col in per_puzzle_prompt_cols[:3]:  # Check first few
            prefix = None
            for p in ["single_model", "aggressive", "positional", "neutral", "moderator", "judge"]:
                if p in prompt_col.lower():
                    prefix = p
                    break
            
            if prefix:
                matching_completion = [c for c in per_puzzle_completion_cols if prefix in c.lower()]
                matching_total = [c for c in per_puzzle_total_cols if prefix in c.lower()]
                
                if matching_completion and matching_total:
                    completion_col = matching_completion[0]
                    total_col = matching_total[0]
                    
                    # Check if total matches sum (only check rows where at least one value is non-zero)
                    mismatches = 0
                    for idx in df.index:
                        total_val = df.loc[idx, total_col]
                        prompt_val = df.loc[idx, prompt_col]
                        completion_val = df.loc[idx, completion_col]
                        if pd.notna(total_val) and pd.notna(prompt_val) and pd.notna(completion_val):
                            # Only check if at least one value is non-zero (skip rows where all are zero)
                            if total_val > 0 or prompt_val > 0 or completion_val > 0:
                                expected_total = prompt_val + completion_val
                                if abs(total_val - expected_total) > 1:  # Allow 1 token difference
                                    mismatches += 1
                    if mismatches > 0:
                        stats["issues"].append(f"{mismatches} rows where {total_col} != {prompt_col} + {completion_col}")
    
    # Check total columns (total_prompt_tokens + total_completion_tokens = total_tokens)
    if prompt_cols and completion_cols:
        total_cols = [col for col in df.columns if col == "total_tokens" or 
                     (col.lower() == "total_tokens" and "prompt" not in col.lower() and "completion" not in col.lower())]
        
        if total_cols:
            for total_col in total_cols[:1]:  # Check first total column
                for prompt_col in prompt_cols[:1]:
                    for completion_col in completion_cols[:1]:
                        # Check if total matches sum (allowing for small rounding differences)
                        mismatches = 0
                        for idx in df.index:
                            total_val = df.loc[idx, total_col]
                            prompt_val = df.loc[idx, prompt_col]
                            completion_val = df.loc[idx, completion_col]
                            if pd.notna(total_val) and pd.notna(prompt_val) and pd.notna(completion_val):
                                expected_total = prompt_val + completion_val
                                if abs(total_val - expected_total) > 1:  # Allow 1 token difference for rounding
                                    mismatches += 1
                        if mismatches > 0:
                            stats["issues"].append(f"{mismatches} rows where {total_col} != {prompt_col} + {completion_col}")
    
    # Count valid rows (no errors)
    if "error" in df.columns:
        if df["error"].dtype == object:
            stats["valid_rows"] = (df["error"] == "").sum()
        else:
            stats["valid_rows"] = df["error"].isna().sum()
    else:
        stats["valid_rows"] = len(df)
    
    return stats


def detect_paradigm(df: pd.DataFrame) -> str:
    """Detect which paradigm was used based on available columns and data."""
    # Check for debate v2 indicators
    if "debate_v2_history" in df.columns and (df["debate_v2_history"] != "").any():
        return "debate_v2"
    
    # Check for self-consistency indicators
    if ("aggressive_move" in df.columns and (df["aggressive_move"] != "").any()) or \
       ("positional_move" in df.columns and (df["positional_move"] != "").any()) or \
       ("neutral_move" in df.columns and (df["neutral_move"] != "").any()) or \
       ("self_consistency_total_prompt_tokens" in df.columns and (df["self_consistency_total_prompt_tokens"] > 0).any()):
        return "self_consistency"
    
    # Check for debate indicators
    if ("debate_history" in df.columns and (df["debate_history"] != "").any()) or \
       ("debate_total_prompt_tokens" in df.columns and (df["debate_total_prompt_tokens"] > 0).any()):
        return "debate"
    
    # Default to single model
    if "single_model_move" in df.columns or "single_model_response" in df.columns:
        return "single_model"
    
    return "unknown"


def calculate_statistics(df: pd.DataFrame, model_name: str) -> Dict[str, any]:
    """Calculate statistics for a model's results."""
    stats = {
        "model": model_name,
        "total_puzzles": len(df),
        "solved": 0,
        "accuracy": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "avg_prompt_tokens_per_puzzle": 0.0,
        "avg_completion_tokens_per_puzzle": 0.0,
        "avg_tokens_per_puzzle": 0.0,
        "paradigm": "unknown",
        "errors": 0,
        "error_rate": 0.0,
    }
    
    # Detect paradigm
    stats["paradigm"] = detect_paradigm(df)
    
    if "puzzle_solved" in df.columns:
        stats["solved"] = df["puzzle_solved"].sum()
        stats["accuracy"] = (stats["solved"] / stats["total_puzzles"] * 100) if stats["total_puzzles"] > 0 else 0.0
    
    # Count errors
    if "error" in df.columns:
        if df["error"].dtype == object:
            stats["errors"] = (df["error"] != "").sum()
        else:
            stats["errors"] = df["error"].notna().sum()
        stats["error_rate"] = (stats["errors"] / stats["total_puzzles"] * 100) if stats["total_puzzles"] > 0 else 0.0
    
    # Find token columns based on paradigm
    paradigm = stats["paradigm"]
    
    if paradigm == "single_model":
        # Use single_model columns
        if "single_model_total_prompt_tokens" in df.columns:
            stats["total_prompt_tokens"] = df["single_model_total_prompt_tokens"].sum()
        elif "single_model_prompt_tokens" in df.columns:
            stats["total_prompt_tokens"] = df["single_model_prompt_tokens"].sum()
        elif "total_prompt_tokens" in df.columns:
            stats["total_prompt_tokens"] = df["total_prompt_tokens"].sum()
        
        if "single_model_total_completion_tokens" in df.columns:
            stats["total_completion_tokens"] = df["single_model_total_completion_tokens"].sum()
        elif "single_model_completion_tokens" in df.columns:
            stats["total_completion_tokens"] = df["single_model_completion_tokens"].sum()
        elif "total_completion_tokens" in df.columns:
            stats["total_completion_tokens"] = df["total_completion_tokens"].sum()
        
        if "single_model_total_tokens" in df.columns:
            stats["total_tokens"] = df["single_model_total_tokens"].sum()
        elif "total_tokens" in df.columns:
            stats["total_tokens"] = df["total_tokens"].sum()
        else:
            stats["total_tokens"] = stats["total_prompt_tokens"] + stats["total_completion_tokens"]
    
    elif paradigm == "self_consistency":
        # Use self_consistency columns
        if "self_consistency_total_prompt_tokens" in df.columns:
            stats["total_prompt_tokens"] = df["self_consistency_total_prompt_tokens"].sum()
        elif "total_prompt_tokens" in df.columns:
            stats["total_prompt_tokens"] = df["total_prompt_tokens"].sum()
        else:
            # Sum individual agent tokens
            for col in ["aggressive_prompt_tokens", "positional_prompt_tokens", "neutral_prompt_tokens"]:
                if col in df.columns:
                    stats["total_prompt_tokens"] += df[col].sum()
        
        if "self_consistency_total_completion_tokens" in df.columns:
            stats["total_completion_tokens"] = df["self_consistency_total_completion_tokens"].sum()
        elif "total_completion_tokens" in df.columns:
            stats["total_completion_tokens"] = df["total_completion_tokens"].sum()
        else:
            # Sum individual agent tokens
            for col in ["aggressive_completion_tokens", "positional_completion_tokens", "neutral_completion_tokens"]:
                if col in df.columns:
                    stats["total_completion_tokens"] += df[col].sum()
        
        if "self_consistency_total_tokens" in df.columns:
            stats["total_tokens"] = df["self_consistency_total_tokens"].sum()
        elif "total_tokens" in df.columns:
            stats["total_tokens"] = df["total_tokens"].sum()
        else:
            stats["total_tokens"] = stats["total_prompt_tokens"] + stats["total_completion_tokens"]
    
    elif paradigm in ["debate", "debate_v2"]:
        # Use debate columns
        if "debate_total_prompt_tokens" in df.columns:
            stats["total_prompt_tokens"] = df["debate_total_prompt_tokens"].sum()
        elif "total_prompt_tokens" in df.columns:
            stats["total_prompt_tokens"] = df["total_prompt_tokens"].sum()
        else:
            # Sum individual agent tokens
            for col in ["aggressive_prompt_tokens", "positional_prompt_tokens", 
                       "moderator_prompt_tokens", "judge_prompt_tokens"]:
                if col in df.columns:
                    stats["total_prompt_tokens"] += df[col].sum()
        
        if "debate_total_completion_tokens" in df.columns:
            stats["total_completion_tokens"] = df["debate_total_completion_tokens"].sum()
        elif "total_completion_tokens" in df.columns:
            stats["total_completion_tokens"] = df["total_completion_tokens"].sum()
        else:
            # Sum individual agent tokens
            for col in ["aggressive_completion_tokens", "positional_completion_tokens",
                       "moderator_completion_tokens", "judge_completion_tokens"]:
                if col in df.columns:
                    stats["total_completion_tokens"] += df[col].sum()
        
        if "debate_total_tokens" in df.columns:
            stats["total_tokens"] = df["debate_total_tokens"].sum()
        elif "total_tokens" in df.columns:
            stats["total_tokens"] = df["total_tokens"].sum()
        else:
            stats["total_tokens"] = stats["total_prompt_tokens"] + stats["total_completion_tokens"]
    
    else:
        # Fallback: try to find any token columns
        if "total_prompt_tokens" in df.columns:
            stats["total_prompt_tokens"] = df["total_prompt_tokens"].sum()
        if "total_completion_tokens" in df.columns:
            stats["total_completion_tokens"] = df["total_completion_tokens"].sum()
        if "total_tokens" in df.columns:
            stats["total_tokens"] = df["total_tokens"].sum()
        else:
            stats["total_tokens"] = stats["total_prompt_tokens"] + stats["total_completion_tokens"]
    
    if stats["total_puzzles"] > 0:
        stats["avg_prompt_tokens_per_puzzle"] = stats["total_prompt_tokens"] / stats["total_puzzles"]
        stats["avg_completion_tokens_per_puzzle"] = stats["total_completion_tokens"] / stats["total_puzzles"]
        stats["avg_tokens_per_puzzle"] = stats["total_tokens"] / stats["total_puzzles"]
    
    return stats


def create_accuracy_plot(all_stats: List[Dict], output_file: str):
    """Create a bar plot comparing accuracy across models."""
    models = [s["model"] for s in all_stats]
    accuracies = [s["accuracy"] for s in all_stats]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(models)), accuracies, color='steelblue', alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Puzzle Solving Accuracy by Model', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy plot to {output_file}")
    plt.close()


def create_token_usage_plot(all_stats: List[Dict], output_file: str):
    """Create a comparison plot of prompt vs completion tokens."""
    models = [s["model"] for s in all_stats]
    prompt_tokens = [s["avg_prompt_tokens_per_puzzle"] for s in all_stats]
    completion_tokens = [s["avg_completion_tokens_per_puzzle"] for s in all_stats]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width/2, prompt_tokens, width, label='Prompt Tokens', color='#2ecc71', alpha=0.7)
    bars2 = ax.bar(x + width/2, completion_tokens, width, label='Completion Tokens', color='#e74c3c', alpha=0.7)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Tokens per Puzzle', fontsize=12)
    ax.set_title('Average Token Usage per Puzzle by Model (Prompt vs Completion)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels for non-zero bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved token usage plot to {output_file}")
    plt.close()


def create_total_tokens_plot(all_stats: List[Dict], output_file: str):
    """Create a plot showing total token usage across models."""
    models = [s["model"] for s in all_stats]
    total_tokens = [s["total_tokens"] / 1000 for s in all_stats]  # Convert to thousands
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(models)), total_tokens, color='purple', alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Total Tokens (thousands)', fontsize=12)
    plt.title('Total Token Usage by Model', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, tokens in zip(bars, total_tokens):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{tokens:.1f}K',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved total tokens plot to {output_file}")
    plt.close()


def create_accuracy_vs_tokens_scatter(all_stats: List[Dict], output_file: str):
    """Create a scatter plot of accuracy vs token efficiency."""
    models = [s["model"] for s in all_stats]
    accuracies = [s["accuracy"] for s in all_stats]
    avg_tokens = [s["avg_tokens_per_puzzle"] for s in all_stats]
    paradigms = [s.get("paradigm", "unknown") for s in all_stats]
    
    # Color by paradigm
    colors_map = {
        "single_model": "#3498db",
        "self_consistency": "#e74c3c",
        "debate": "#9b59b6",
        "debate_v2": "#9b59b6",
        "unknown": "#95a5a6"
    }
    scatter_colors = [colors_map.get(p, "#95a5a6") for p in paradigms]
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(avg_tokens, accuracies, s=200, alpha=0.6, c=scatter_colors)
    
    # Add model labels (only for models with accuracy > 0 or tokens > 0)
    for i, model in enumerate(models):
        if accuracies[i] > 0 or avg_tokens[i] > 0:
            plt.annotate(model, (avg_tokens[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Average Tokens per Puzzle', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Token Efficiency by Paradigm', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[p], label=p.replace('_', ' ').title()) 
                      for p in set(paradigms)]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy vs tokens scatter plot to {output_file}")
    plt.close()


def create_error_analysis_plot(all_stats: List[Dict], output_file: str):
    """Create a plot showing error rates across models."""
    models = [s["model"] for s in all_stats]
    error_rates = [s.get("error_rate", 0.0) for s in all_stats]
    paradigms = [s.get("paradigm", "unknown") for s in all_stats]
    
    # Color by paradigm
    colors = {
        "single_model": "#3498db",
        "self_consistency": "#e74c3c",
        "debate": "#9b59b6",
        "debate_v2": "#9b59b6",
        "unknown": "#95a5a6"
    }
    bar_colors = [colors.get(p, "#95a5a6") for p in paradigms]
    
    plt.figure(figsize=(16, 8))
    bars = plt.bar(range(len(models)), error_rates, color=bar_colors, alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.title('Error Rate by Model and Paradigm', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[p], label=p.replace('_', ' ').title()) 
                      for p in set(paradigms)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, error_rates)):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved error analysis plot to {output_file}")
    plt.close()


def print_validation_report(all_validations: List[Dict], all_stats: List[Dict]):
    """Print a validation report."""
    print("\n" + "="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)
    
    # Create a mapping from model to stats for paradigm info
    stats_map = {s['model']: s for s in all_stats}
    
    for val in all_validations:
        model_name = val['model']
        stats = stats_map.get(model_name, {})
        paradigm = stats.get('paradigm', 'unknown')
        
        print(f"\nModel: {model_name} ({paradigm})")
        print(f"  Total puzzles: {val['total_puzzles']}")
        print(f"  Valid rows: {val['valid_rows']}")
        print(f"  Errors: {stats.get('errors', 0)} ({stats.get('error_rate', 0.0):.1f}%)")
        
        if val['issues']:
            print(f"  ⚠️  Issues found:")
            for issue in val['issues']:
                print(f"     - {issue}")
        else:
            print(f"  ✅ No critical issues found")


def print_statistics_summary(all_stats: List[Dict]):
    """Print a summary of statistics."""
    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)
    
    # Create DataFrame for easier display
    df_summary = pd.DataFrame(all_stats)
    
    # Sort by accuracy descending
    df_summary = df_summary.sort_values('accuracy', ascending=False)
    
    print("\nModel Performance:")
    print("-" * 100)
    print(f"{'Model':<35} {'Paradigm':<15} {'Solved':<10} {'Accuracy':<12} {'Errors':<10} {'Avg Tokens/Puzzle':<20}")
    print("-" * 100)
    
    for _, row in df_summary.iterrows():
        paradigm = row.get('paradigm', 'unknown')
        errors = int(row.get('errors', 0))
        print(f"{row['model']:<35} {paradigm:<15} {int(row['solved']):<10} {row['accuracy']:>6.2f}%     {errors:<10} {row['avg_tokens_per_puzzle']:>10.0f}")
    
    print("\nToken Usage Summary:")
    print("-" * 100)
    print(f"{'Model':<35} {'Paradigm':<15} {'Total Prompt':<15} {'Total Completion':<18} {'Total Tokens':<15}")
    print("-" * 100)
    
    for _, row in df_summary.iterrows():
        paradigm = row.get('paradigm', 'unknown')
        print(f"{row['model']:<35} {paradigm:<15} {row['total_prompt_tokens']:>12,}    {row['total_completion_tokens']:>15,}    {row['total_tokens']:>12,}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and validate chess puzzle evaluation results")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Directory containing CSV result files (default: chess_puzzles/data)")
    parser.add_argument("--games-dir", type=str, default=None,
                       help="Directory containing JSON game results (default: chess_puzzles/data/games)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save plots (default: chess_puzzles/graphs)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots, only validate and print statistics")
    parser.add_argument("--combined-output", type=str, default=None,
                       help="Path to save combined summary CSV (default: graphs/summary_statistics.csv)")
    parser.add_argument("--selected-models", type=str, default=None,
                       help="Comma-separated list of models to include (default: all)")
    
    args = parser.parse_args()
    
    # Set default directories
    if args.data_dir is None:
        args.data_dir = os.path.join(parent_dir, "data")
    if args.games_dir is None:
        args.games_dir = os.path.join(parent_dir, "data", "games")
    if args.output_dir is None:
        args.output_dir = os.path.join(parent_dir, "graphs")
    if args.combined_output is None:
        args.combined_output = os.path.join(args.output_dir, "summary_statistics.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading CSV files from: {args.data_dir}")
    csv_files = load_csv_files(args.data_dir)
    
    if args.selected_models:
        selected = set([name.strip() for name in args.selected_models.split(",") if name.strip()])
        csv_files = {k: v for k, v in csv_files.items() if k in selected}
    
    if not csv_files:
        print("No CSV result files found!")
    else:
        print(f"\nFound {len(csv_files)} result file(s)")
    
    # Validate and calculate statistics
    all_validations = []
    all_stats = []
    
    for model_name, df in csv_files.items():
        print(f"\nProcessing {model_name}...")
        validation = validate_data(df, model_name)
        all_validations.append(validation)
        
        stats = calculate_statistics(df, model_name)
        all_stats.append(stats)
        
        # Additional validation: check prompt_token_log if available
        if "prompt_token_log" in df.columns or "single_model_prompt_log" in df.columns:
            log_cols = [col for col in df.columns if "prompt_log" in col.lower() or "token_log" in col.lower()]
            for log_col in log_cols:
                non_empty_logs = df[df[log_col].notna() & (df[log_col] != "") & (df[log_col] != "[]")]
                if len(non_empty_logs) > 0:
                    # Try to parse JSON and validate token counts
                    try:
                        for idx, log_str in non_empty_logs[log_col].items():
                            if log_str and log_str != "[]":
                                log_data = json.loads(log_str) if isinstance(log_str, str) else log_str
                                if isinstance(log_data, list):
                                    for event in log_data:
                                        if isinstance(event, dict):
                                            prompt_t = event.get("prompt_tokens", 0) or 0
                                            completion_t = event.get("completion_tokens", 0) or 0
                                            total_t = event.get("total_tokens", 0) or 0
                                            expected_total = prompt_t + completion_t
                                            if abs(total_t - expected_total) > 1:
                                                validation["issues"].append(
                                                    f"Token log inconsistency in {log_col} at row {idx}: "
                                                    f"total_tokens ({total_t}) != prompt_tokens ({prompt_t}) + completion_tokens ({completion_t})"
                                                )
                    except (json.JSONDecodeError, TypeError) as e:
                        # Log parsing errors are not critical
                        pass
    
    # Print reports
    if csv_files:
        print_validation_report(all_validations, all_stats)
        print_statistics_summary(all_stats)
    
        # Generate plots
        if not args.no_plots:
            print("\n" + "="*80)
            print("GENERATING PUZZLE PLOTS")
            print("="*80)
            
            create_accuracy_plot(all_stats, os.path.join(args.output_dir, "accuracy_by_model.png"))
            create_token_usage_plot(all_stats, os.path.join(args.output_dir, "token_usage_by_model.png"))
            create_total_tokens_plot(all_stats, os.path.join(args.output_dir, "total_tokens_by_model.png"))
            create_accuracy_vs_tokens_scatter(all_stats, os.path.join(args.output_dir, "accuracy_vs_tokens.png"))
            create_error_analysis_plot(all_stats, os.path.join(args.output_dir, "error_rates_by_model.png"))
            
            print(f"\n✅ Puzzle plots saved to {args.output_dir}")
        
        # Save summary CSV
        summary_df = pd.DataFrame(all_stats)
        summary_df.to_csv(args.combined_output, index=False)
        print(f"\n✅ Puzzle summary statistics saved to {args.combined_output}")
    
    # Process full game results if any
    game_entries = load_game_results(args.games_dir)
    if game_entries:
        game_stats = [analyze_game_result(entry) for entry in game_entries]
        print_game_summary(game_stats)
        
        if not args.no_plots:
            print("\n" + "="*80)
            print("GENERATING GAME PLOTS")
            print("="*80)
            # Potential additional game-specific plots can be added here
        
        # Save game summary CSV
        game_summary_df = pd.DataFrame(game_stats)
        game_summary_file = os.path.join(args.output_dir, "game_summary_statistics.csv")
        game_summary_df.to_csv(game_summary_file, index=False)
        print(f"\n✅ Game summary statistics saved to {game_summary_file}")


if __name__ == "__main__":
    main()

