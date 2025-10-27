#!/bin/bash

# Chess Game Engine Runner Script
# Quick commands to run different game configurations

echo "🎮 Chess Game Engine - Quick Runner"
echo "=================================="

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Check if Stockfish is available
if ! command -v stockfish &> /dev/null; then
    echo "❌ Error: Stockfish not found in PATH"
    echo "Please install Stockfish or specify path with --stockfish"
    exit 1
fi

echo "✅ Environment check passed"
echo ""

# Function to run a game
run_game() {
    local description="$1"
    local args="$2"
    
    echo "🎯 $description"
    echo "Command: python chess_game.py $args"
    echo ""
    
    python chess_game.py $args
    
    echo ""
    echo "=================================="
    echo ""
}

# Menu
echo "Choose a game configuration:"
echo "1. Single model vs Stockfish (White, save PGN)"
echo "2. Self-consistency vs Stockfish (White, save PGN)"
echo "3. Single model vs Stockfish (Black, save PGN)"
echo "4. Self-consistency vs Stockfish (Black, save PGN)"
echo "5. Single model vs Random (White, save PGN)"
echo "6. Self-consistency vs Random (White, save PGN)"
echo "7. Quick single model game (no save)"
echo "8. Quick self-consistency game (no save)"
echo "9. Custom game"
echo ""

read -p "Enter choice (1-9): " choice

case $choice in
    1)
        run_game "Single Model vs Stockfish (Model plays White)" "--save-pgn"
        ;;
    2)
        run_game "Self-Consistency vs Stockfish (Model plays White)" "--self-consistency --save-pgn"
        ;;
    3)
        run_game "Single Model vs Stockfish (Model plays Black)" "--model-color black --save-pgn"
        ;;
    4)
        run_game "Self-Consistency vs Stockfish (Model plays Black)" "--self-consistency --model-color black --save-pgn"
        ;;
    5)
        run_game "Single Model vs Random (Model plays White)" "--random-opponent --save-pgn"
        ;;
    6)
        run_game "Self-Consistency vs Random (Model plays White)" "--self-consistency --random-opponent --save-pgn"
        ;;
    7)
        run_game "Quick Single Model Game" ""
        ;;
    8)
        run_game "Quick Self-Consistency Game" "--self-consistency"
        ;;
    9)
        echo "Custom game options:"
        echo "--model MODEL_NAME (e.g., gpt-4-turbo)"
        echo "--time SECONDS (Stockfish thinking time)"
        echo "--self-consistency (use self-consistency)"
        echo "--random-opponent (use random legal moves instead of Stockfish)"
        echo "--model-color white|black"
        echo "--save-json (save as JSON)"
        echo "--save-pgn (save as PGN)"
        echo ""
        read -p "Enter custom arguments: " custom_args
        run_game "Custom Game" "$custom_args"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "🎉 Game completed!"

