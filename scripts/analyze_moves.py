#!/usr/bin/env python3
"""
Analyze correct moves vs total moves from puzzle evaluation results.
"""

import pandas as pd
import sys
from pathlib import Path

def analyze_moves(csv_file: str):
    """Analyze correct moves vs total moves from CSV results."""
    df = pd.read_csv(csv_file)
    
    print("="*80)
    print(f"ANALYZING: {Path(csv_file).name}")
    print("="*80)
    print()
    
    # Basic statistics
    total_puzzles = len(df)
    if "puzzle_solved" in df.columns:
        solved = df["puzzle_solved"].sum()
        accuracy = (solved / total_puzzles * 100) if total_puzzles > 0 else 0
        print(f"Total puzzles: {total_puzzles}")
        print(f"Puzzles solved: {solved}")
        print(f"Accuracy: {accuracy:.2f}%")
        print()
    
    # Analyze correct_moves vs total_moves
    if "correct_moves" in df.columns and "total_moves" in df.columns:
        print("="*80)
        print("CORRECT MOVES vs TOTAL MOVES ANALYSIS")
        print("="*80)
        print()
        
        # Convert to numeric, handling any non-numeric values
        df["correct_moves"] = pd.to_numeric(df["correct_moves"], errors="coerce")
        df["total_moves"] = pd.to_numeric(df["total_moves"], errors="coerce")
        
        # Overall statistics
        total_correct = df["correct_moves"].sum()
        total_required = df["total_moves"].sum()
        avg_correct = df["correct_moves"].mean()
        avg_total = df["total_moves"].mean()
        overall_move_accuracy = (total_correct / total_required * 100) if total_required > 0 else 0
        
        print(f"Total correct moves across all puzzles: {total_correct:.0f}")
        print(f"Total moves required across all puzzles: {total_required:.0f}")
        print(f"Overall move accuracy: {overall_move_accuracy:.2f}% ({total_correct:.0f}/{total_required:.0f})")
        print(f"Average correct moves per puzzle: {avg_correct:.2f}")
        print(f"Average total moves per puzzle: {avg_total:.2f}")
        print()
        
        # Breakdown by correct moves count
        print("Breakdown by number of correct moves:")
        print("-" * 80)
        move_counts = df["correct_moves"].value_counts().sort_index(ascending=False)
        for moves, count in move_counts.items():
            percentage = (count / total_puzzles * 100) if total_puzzles > 0 else 0
            # Calculate average total moves for puzzles with this many correct moves
            puzzles_with_moves = df[df["correct_moves"] == moves]
            avg_total_for_this = puzzles_with_moves["total_moves"].mean() if len(puzzles_with_moves) > 0 else 0
            move_accuracy = (moves / avg_total_for_this * 100) if avg_total_for_this > 0 else 0
            print(f"  {int(moves)} correct moves: {count:3d} puzzles ({percentage:5.1f}%) | Avg total: {avg_total_for_this:.1f} | Move accuracy: {move_accuracy:.1f}%")
        print()
        
        # Puzzles with partial success (some correct moves but not solved)
        if "puzzle_solved" in df.columns:
            partial = df[(df["correct_moves"] > 0) & (df["puzzle_solved"] == False)]
            if len(partial) > 0:
                partial_pct = (len(partial) / total_puzzles * 100) if total_puzzles > 0 else 0
                avg_correct_partial = partial['correct_moves'].mean()
                avg_total_partial = partial['total_moves'].mean()
                partial_accuracy = (avg_correct_partial / avg_total_partial * 100) if avg_total_partial > 0 else 0
                print(f"Puzzles with partial success (correct moves > 0 but not solved): {len(partial)} ({partial_pct:.1f}%)")
                print(f"  Average correct moves in partial: {avg_correct_partial:.2f}")
                print(f"  Average total moves in partial: {avg_total_partial:.2f}")
                print(f"  Average move accuracy in partial: {partial_accuracy:.1f}%")
                print()
        
        # Distribution of correct_moves / total_moves ratio
        df["move_ratio"] = df["correct_moves"] / df["total_moves"]
        df["move_ratio"] = df["move_ratio"].replace([float('inf'), float('-inf')], pd.NA)
        
        print("Move accuracy distribution (correct_moves / total_moves):")
        print("-" * 80)
        ratio_ranges = [
            (1.0, "100% (fully solved)"),
            (0.5, 1.0, "50-99% (mostly correct)"),
            (0.1, 0.5, "10-49% (partially correct)"),
            (0.0, 0.1, "0-9% (mostly wrong)"),
            (0.0, "0% (completely wrong)"),
        ]
        
        for range_def in ratio_ranges:
            if len(range_def) == 2:
                # Single value (exact match)
                if range_def[0] == 1.0:
                    count = (df["move_ratio"] == 1.0).sum()
                else:  # 0.0
                    count = (df["move_ratio"] == 0.0).sum()
                label = range_def[1]
            else:
                # Range
                min_val, max_val, label = range_def
                count = ((df["move_ratio"] >= min_val) & (df["move_ratio"] < max_val)).sum()
            
            percentage = (count / total_puzzles * 100) if total_puzzles > 0 else 0
            # Calculate cumulative percentage
            if range_def[0] == 1.0:
                cum_pct = percentage
            elif range_def[0] == 0.0 and len(range_def) == 2:
                cum_pct = percentage
            else:
                # For ranges, calculate cumulative from top
                if range_def[0] >= 0.5:
                    cum_count = (df["move_ratio"] >= range_def[0]).sum()
                else:
                    cum_count = count
                cum_pct = (cum_count / total_puzzles * 100) if total_puzzles > 0 else 0
            
            print(f"  {label:30s}: {count:3d} puzzles ({percentage:5.1f}%) | Cumulative: {cum_pct:5.1f}%")
        print()
        
        # Show examples of puzzles with different move counts
        print("="*80)
        print("SAMPLE PUZZLES")
        print("="*80)
        print()
        
        # Fully solved puzzles
        if "puzzle_solved" in df.columns:
            solved_puzzles = df[df["puzzle_solved"] == True]
            if len(solved_puzzles) > 0:
                print(f"Fully solved puzzles (showing up to 5):")
                sample_cols = ["PuzzleId", "correct_moves", "total_moves", "single_model_move", "Moves"]
                available_cols = [col for col in sample_cols if col in df.columns]
                print(solved_puzzles[available_cols].head(5).to_string(index=False))
                print()
        
        # Puzzles with partial success
        partial = df[(df["correct_moves"] > 0) & (df["puzzle_solved"] == False)]
        if len(partial) > 0:
            print(f"Puzzles with partial success (showing up to 5):")
            sample_cols = ["PuzzleId", "correct_moves", "total_moves", "single_model_move", "Moves"]
            available_cols = [col for col in sample_cols if col in df.columns]
            print(partial[available_cols].head(5).to_string(index=False))
            print()
        
        # Puzzles with 0 correct moves
        zero_correct = df[df["correct_moves"] == 0]
        if len(zero_correct) > 0:
            print(f"Puzzles with 0 correct moves (showing up to 5):")
            sample_cols = ["PuzzleId", "correct_moves", "total_moves", "single_model_move", "Moves"]
            available_cols = [col for col in sample_cols if col in df.columns]
            print(zero_correct[available_cols].head(5).to_string(index=False))
            print()
    else:
        print("⚠️  Warning: 'correct_moves' or 'total_moves' columns not found in CSV")
        print(f"Available columns: {', '.join(df.columns.tolist())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_moves.py <csv_file>")
        print("\nExample:")
        print("  python3 analyze_moves.py data/test_results/test_results_qwen_qwen3-235b-a22b-instruct-2507_single_50.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    analyze_moves(csv_file)

