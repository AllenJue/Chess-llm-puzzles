#!/bin/bash

# Script to run all three systems (single model, self-consistency, debate) on puzzles 50-100
# This avoids re-running the first 50 puzzles

echo "🚀 Starting evaluation on puzzles 50-100 (puzzle indices 50-99)..."
echo "📊 This will run all three systems:"
echo "   1. Single model (baseline)"
echo "   2. Self-consistency system (3 independent queries with majority vote)"
echo "   3. Debate system (aggressive vs positional + moderator + judge)"
echo ""

# Create timestamped output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SINGLE_OUTPUT="results_single_model_50_100_${TIMESTAMP}.csv"
SELF_CONSISTENCY_OUTPUT="results_self_consistency_50_100_${TIMESTAMP}.csv"
DEBATE_OUTPUT="results_debate_50_100_${TIMESTAMP}.csv"

echo "📁 Output files will be:"
echo "   - Single model: ${SINGLE_OUTPUT}"
echo "   - Self-consistency: ${SELF_CONSISTENCY_OUTPUT}"
echo "   - Debate: ${DEBATE_OUTPUT}"
echo "⏰ Started at: $(date)"
echo "============================================================"

# Function to run evaluation
run_evaluation() {
    local system_name="$1"
    local output_file="$2"
    local extra_args="$3"
    
    echo ""
    echo "🔄 Running ${system_name} evaluation..."
    echo "   Output: ${output_file}"
    
    python main.py \
        --evaluate \
        ${extra_args} \
        --max-puzzles 50 \
        --start-puzzle 50 \
        --csv-file ./data/lichess_puzzles_with_pgn_1000.csv \
        --output "${output_file}"
    
    if [ $? -eq 0 ]; then
        echo "✅ ${system_name} completed successfully!"
        if [ -f "${output_file}" ]; then
            FILE_SIZE=$(stat -f%z "${output_file}" 2>/dev/null || stat -c%s "${output_file}" 2>/dev/null)
            echo "   📏 File size: ${FILE_SIZE} bytes"
        fi
    else
        echo "❌ ${system_name} failed!"
        return 1
    fi
}

# Run all three evaluations
echo "1️⃣ Running single model evaluation..."
run_evaluation "Single Model" "${SINGLE_OUTPUT}" ""

echo ""
echo "2️⃣ Running self-consistency system evaluation..."
run_evaluation "Self-Consistency System" "${SELF_CONSISTENCY_OUTPUT}" "--self-consistency"

echo ""
echo "3️⃣ Running debate system evaluation..."
run_evaluation "Debate System" "${DEBATE_OUTPUT}" "--debate"

echo ""
echo "============================================================"
echo "🎉 All evaluations completed!"
echo "⏰ Finished at: $(date)"
echo ""
echo "📊 Results summary:"
echo "   - Single model: ${SINGLE_OUTPUT}"
echo "   - Self-consistency: ${SELF_CONSISTENCY_OUTPUT}"
echo "   - Debate: ${DEBATE_OUTPUT}"
echo ""
echo "🔍 You can now compare the performance of all three systems on puzzles 50-100!"
