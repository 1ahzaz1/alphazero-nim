import os
import re
import numpy as np
import torch
from game import Nim
from mcts import MCTS
from model import Nim_Model
from main import set_seed
import matplotlib.pyplot as plt
from tabulate import tabulate  # Install with: pip install tabulate
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model generalization across different board positions")
    parser.add_argument("--history", type=int, choices=[0, 1], default=1,
                        help="Use history in analysis (0=False, 1=True)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Alpha value used during training (for directory path)")
    parser.add_argument("--sims", type=int, default=100,
                        help="Number of MCTS simulations for analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint to use (e.g., 500 for checkpoint_iter_500.pt)")
    return parser.parse_args()

# Get command line arguments
args = parse_args()

# --- CONFIG (parameters can be overridden by command line args) ---
include_history = bool(args.history)
num_frames = 2
num_simulation = args.sims  # Use fewer simulations to highlight network differences
model_base_dir = './models'
training_alpha = args.alpha  # Alpha used during training (for directory path)
analysis_alpha = 0.8  # Higher alpha for analysis
analysis_epsilon = 0.5  # Force exploration during analysis
temperature = 1.0  # Higher temperature for more balanced output
checkpoint = args.checkpoint  # Set to specific checkpoint number or None for latest
# -------------------------------------

set_seed(30)

# Define test boards with 5 piles across different difficulty levels
# All boards have 5 piles to match training input size
# IMPORTANT: All boards must have the same number of total stones (25) to ensure same action space

# Level 1: Simple positions similar to training (total stones = 25)
simple_positions = [
    # Training board (base position)
    [1, 3, 5, 7, 9],  # nim-sum = 1 (N-position)
    # Simple variations (same total stones)
    [2, 4, 4, 6, 9],  # nim-sum = 1 (N-position)
    [3, 3, 4, 6, 9],  # nim-sum = 9 (N-position)
]

# Level 2: Medium difficulty - different structures (total stones = 25)
medium_positions = [
    [1, 1, 3, 8, 12],  # nim-sum = 5 (N-position) 
    [5, 5, 5, 5, 5],   # nim-sum = 5 (N-position)
    [2, 2, 5, 7, 9],   # nim-sum = 3 (N-position)
]

# Level 3: Hard - complex positions (total stones = 25)
hard_positions = [
    [1, 2, 3, 9, 10],  # nim-sum = 3 (N-position)
    [1, 1, 1, 8, 14],  # nim-sum = 7 (N-position)
    [3, 4, 6, 6, 6]    # nim-sum = 3 (N-position)
]

# Level 4: Extreme - asymmetric positions (total stones = 25)
extreme_positions = [
    [1, 1, 1, 1, 21],   # nim-sum = 21 (N-position)
    [1, 1, 1, 2, 20],   # nim-sum = 21 (N-position)
    [1, 2, 2, 5, 15]    # nim-sum = 9 (N-position)
]

def calculate_nim_sum(board):
    """Calculate the nim-sum of a board"""
    xor = 0
    for c in board:
        xor = c ^ xor
    return xor

def get_optimal_moves(game, board):
    """Get all optimal moves for a board (moves that lead to nim-sum = 0)"""
    optimal_moves = []
    xor = calculate_nim_sum(board)
    
    if xor == 0:
        # All moves lead to nim-sum ≠ 0, so none are optimal
        return []
    
    # Check each possible action
    for action in game.legal_actions():
        pile_idx, take = game.unpack_action(action)
        new_board = board.copy()
        new_board[pile_idx] -= take
        
        # If this move leads to nim-sum = 0, it's optimal
        if calculate_nim_sum(new_board) == 0:
            optimal_moves.append(action)
    
    return optimal_moves

def analyze_board(board, model_path, device, verbose=True, game_result=None):
    """Analyze model performance on a specific board"""
    if verbose:
        print(f"\n{'='*50}")
        print(f"BOARD: {board} (nim-sum = {calculate_nim_sum(board)})")
        print(f"{'='*50}")
    
    # Initialize game with board
    game = Nim(board, include_history, num_frames=num_frames)
    
    # Initialize a model specifically for this board
    model = Nim_Model(
        action_size=game.action_size,
        input_size=len(board),
        hidden_size=128,
        num_lstm_layers=1,
        num_head_layers=2
    )
    
    # Load weights from the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run MCTS on board state
    mcts = MCTS(game, model, {
        'num_simulations': num_simulation, 
        'alpha': analysis_alpha,
        'epsilon': analysis_epsilon
    })
    state = game.state()
    root = mcts.run(state, game.to_play(), is_train=True)
    
    # Get optimal moves
    optimal_moves = get_optimal_moves(game, board)
    
    # Display root stats
    _, value = model.predict(root.state)
    board_position = "P-position" if calculate_nim_sum(board) == 0 else "N-position"
    if verbose:
        print(f'Board position: {board_position}')
        print(f'Value prediction: {round(value, 2)} (should be negative for P-position, positive for N-position)')
    
    # Get top moves by visit count
    total_visits = sum(child.visit_count for child in root.children.values())
    sorted_children = sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True)
    
    # Calculate metrics
    top_move = sorted_children[0][0] if sorted_children else None
    top_move_optimal = top_move in optimal_moves if top_move else False
    optimal_visit_pct = sum(child.visit_count for action, child in root.children.items() 
                            if action in optimal_moves) / total_visits if total_visits > 0 else 0
    
    # Print metrics
    if verbose:
        print(f"\nOptimal moves: {[game.unpack_action(m) for m in optimal_moves]}")
        print(f"Top move: {game.unpack_action(top_move) if top_move else 'None'} (optimal: {top_move_optimal})")
        print(f"Percentage of visits to optimal moves: {round(optimal_visit_pct * 100, 2)}%")
    
        # Print top 5 moves
        print("\nTop moves:")
        for i, (action, child) in enumerate(sorted_children[:5]):
            if child.state is not None:
                child_board = child.state[len(board)*num_frames:] if include_history else child.state
                predicted_value = model.predict(child.state)[1]
                is_optimal = "OPTIMAL" if action in optimal_moves else "SUBOPTIMAL"
                is_nim_zero = calculate_nim_sum(child_board) == 0
                
                print(f"{i+1}. {game.unpack_action(action)}: {child_board} ({is_optimal}) | "
                      f"Nim-sum = {'0' if is_nim_zero else 'nonzero'} | "
                      f"P: {round(child.prior, 2)} | "
                      f"V: {round(-predicted_value, 2)} | "
                      f"N: {child.visit_count} ({round(child.visit_count/total_visits * 100, 2)}%) | "
                      f"Q: {round(-child.value(), 2)}")
    
    # Return metrics for summary
    value_correct = (value < 0 and board_position == "P-position") or (value > 0 and board_position == "N-position")
    
    results = {
        "board": board,
        "nim_sum": calculate_nim_sum(board),
        "position_type": board_position,
        "value_prediction": value,
        "value_correct": value_correct,
        "top_move_optimal": top_move_optimal,
        "optimal_visit_pct": optimal_visit_pct,
        "total_visits": total_visits
    }
    
    if game_result is not None:
        game_result.append(results)
    
    return results

def analyze_batch(boards, name, model_path, device, all_results):
    """Analyze a batch of boards with the same difficulty level"""
    print(f"\n\n--- ANALYZING {name} POSITIONS ---")
    
    batch_results = []
    for board in boards:
        result = analyze_board(board, model_path, device, verbose=True, game_result=all_results)
        batch_results.append(result)
    
    # Calculate batch metrics
    optimal_moves_pct = sum(r["top_move_optimal"] for r in batch_results) / len(batch_results) * 100
    value_correct_pct = sum(r["value_correct"] for r in batch_results) / len(batch_results) * 100
    
    print(f"\n{name} SUMMARY:")
    print(f"Optimal moves: {optimal_moves_pct:.2f}%")
    print(f"Correct value assessments: {value_correct_pct:.2f}%")
    
    return {
        "name": name,
        "optimal_moves_pct": optimal_moves_pct,
        "value_correct_pct": value_correct_pct,
        "boards": batch_results
    }

def verify_consistent_action_space(boards):
    """Verify that all boards have the same action space size"""
    action_sizes = []
    for board in boards:
        game = Nim(board, include_history, num_frames=num_frames)
        action_sizes.append(len(game.all_legal_actions))
    
    if len(set(action_sizes)) > 1:
        print("\n⚠️ WARNING: Inconsistent action space sizes detected!")
        print("This will cause model loading errors during analysis.")
        print("Boards must have the same total number of stones for consistent action spaces.")
        print("\nAction space sizes:")
        for i, (board, size) in enumerate(zip(boards, action_sizes)):
            stone_sum = sum(board)
            print(f"Board {i+1}: {board} → {size} actions (total stones: {stone_sum})")
        return False, action_sizes[0] if action_sizes else 0
    
    return True, action_sizes[0] if action_sizes else 0

def main():
    # Load the model from training
    train_board = [1, 3, 5, 7, 9]  # Training board
    model_folder = f"{len(train_board)}_{include_history}_{training_alpha}"
    model_dir = os.path.join(model_base_dir, model_folder)
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Please check:")
        print(f"1. Training board size: {len(train_board)}")
        print(f"2. History setting: {include_history}")
        print(f"3. Alpha value: {training_alpha}")
        print("\nAvailable model directories:")
        if os.path.exists(model_base_dir):
            print(os.listdir(model_base_dir))
        else:
            print(f"Base directory {model_base_dir} does not exist")
        return
    
    # Verify all boards have consistent action spaces
    print("Verifying action space consistency...")
    all_boards = simple_positions + medium_positions + hard_positions + extreme_positions
    consistent, action_size = verify_consistent_action_space(all_boards + [train_board])
    if not consistent:
        print("\n❌ Cannot proceed with analysis due to inconsistent action spaces.")
        print("Please modify the board configurations to ensure all have the same number of total stones.")
        print(f"The training board {train_board} has {action_size} actions.")
        return
    
    # Find checkpoint
    if checkpoint:
        model_path = os.path.join(model_dir, f"checkpoint_iter_{checkpoint}.pt")
    else:
        checkpoints = [f for f in os.listdir(model_dir) if re.match(r'checkpoint_iter_\d+\.pt', f)]
        latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
        model_path = os.path.join(model_dir, latest_checkpoint)
    
    print(f"Using model: {model_path}")
    print("History enabled" if include_history else "History disabled")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run analysis on all boards
    all_results = []
    curriculum_results = []
    
    # Analyze each difficulty level
    simple_results = analyze_batch(simple_positions, "SIMPLE", model_path, device, all_results)
    medium_results = analyze_batch(medium_positions, "MEDIUM", model_path, device, all_results)
    hard_results = analyze_batch(hard_positions, "HARD", model_path, device, all_results)
    extreme_results = analyze_batch(extreme_positions, "EXTREME", model_path, device, all_results)
    
    curriculum_results = [simple_results, medium_results, hard_results, extreme_results]
    
    # Print curriculum summary
    print("\n\n" + "="*80)
    print("CURRICULUM GENERALIZATION SUMMARY")
    print("="*80)
    
    # Create table data
    table_data = []
    for result in curriculum_results:
        table_data.append([
            result["name"],
            f"{result['optimal_moves_pct']:.2f}%",
            f"{result['value_correct_pct']:.2f}%"
        ])
    
    # Print table
    table_headers = ["Difficulty", "Optimal Moves", "Correct Value Assessment"]
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    
    # Print overall statistics
    p_optimal = [r["top_move_optimal"] for r in all_results if r["position_type"] == "P-position"]
    n_optimal = [r["top_move_optimal"] for r in all_results if r["position_type"] == "N-position"]
    
    p_optimal_pct = sum(p_optimal) / len(p_optimal) * 100 if p_optimal else 0
    n_optimal_pct = sum(n_optimal) / len(n_optimal) * 100 if n_optimal else 0
    overall_optimal_pct = sum(r["top_move_optimal"] for r in all_results) / len(all_results) * 100
    
    print("\nPOSITION TYPE ANALYSIS:")
    print(f"P-positions optimal play: {p_optimal_pct:.2f}% ({sum(p_optimal)}/{len(p_optimal)})")
    print(f"N-positions optimal play: {n_optimal_pct:.2f}% ({sum(n_optimal)}/{len(n_optimal)})")
    print(f"Overall optimal play: {overall_optimal_pct:.2f}% ({sum(r['top_move_optimal'] for r in all_results)}/{len(all_results)})")
    
    # Generate visualizations if matplotlib is available
    try:
        # Difficulty progression chart
        difficulties = [r["name"] for r in curriculum_results]
        optimal_moves = [r["optimal_moves_pct"] for r in curriculum_results]
        value_correct = [r["value_correct_pct"] for r in curriculum_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(difficulties, optimal_moves, 'bo-', label='Optimal Moves (%)')
        plt.plot(difficulties, value_correct, 'ro-', label='Correct Value Assessment (%)')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Performance (%)')
        plt.title('Model Performance Across Difficulty Levels')
        plt.ylim(0, 110)
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plt.savefig(f'generalization_{include_history}_{len(train_board)}.png')
        print("\nVisualization saved as generalization_[history]_[board_size].png")
    except Exception as e:
        print(f"Could not generate visualization: {e}")

if __name__ == "__main__":
    main() 