import argparse
import torch
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from game import Nim
from mcts import MCTS
from model import Nim_Model


def parse_args():
    parser = argparse.ArgumentParser(description="Tournament between history and no-history models")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to play per board configuration")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Alpha value used in model training")
    parser.add_argument("--simulations", type=int, default=100,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint to use (e.g., 500 for checkpoint_iter_500.pt)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for move selection (0=best move always)")
    parser.add_argument("--boards", type=str, default=None,
                        help="Comma-separated list of board indices to test (e.g., 0,4,10)")
    parser.add_argument("--exploration", type=float, default=0.0,
                        help="Exploration parameter for MCTS during play (alpha value)")
    return parser.parse_args()


def load_model(board, include_history, alpha, checkpoint=None, device=None):
    """Load a trained model with specific configuration"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize game and model
    game = Nim(board, include_history=include_history, num_frames=2)
    model = Nim_Model(
        action_size=game.action_size,
        input_size=len(board),
        hidden_size=128,
        num_lstm_layers=1,
        num_head_layers=2
    ).to(device)
    
    # Determine model path
    model_folder = f"{len(board)}_{include_history}_{alpha}"
    model_dir = f"./models/{model_folder}"
    
    if checkpoint:
        model_path = f"{model_dir}/checkpoint_iter_{checkpoint}.pt"
    else:
        # Find latest checkpoint
        import os
        import re
        checkpoints = [f for f in os.listdir(model_dir) if re.match(r'checkpoint_iter_\d+\.pt', f)]
        latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
        model_path = f"{model_dir}/{latest_checkpoint}"
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return game, model


def play_game(game, model1, model2, model1_first=True, num_simulations=100, temperature=0.0, exploration=0.0):
    """Play a complete game between two models"""
    # Reset game state
    state = game.reset()
    done = False
    
    # Set up MCTS agents
    mcts1 = MCTS(game, model1, {'num_simulations': num_simulations, 'alpha': exploration, 'epsilon': exploration/2})
    mcts2 = MCTS(game, model2, {'num_simulations': num_simulations, 'alpha': exploration, 'epsilon': exploration/2})
    
    # Track game trajectory
    trajectory = []
    
    # Main game loop
    turn = 0
    while not done:
        # Determine whose turn it is
        is_model1_turn = (turn % 2 == 0) if model1_first else (turn % 2 == 1)
        current_mcts = mcts1 if is_model1_turn else mcts2
        
        # Run MCTS and select action
        root = current_mcts.run(state, game.to_play(), is_train=False)
        action = root.select_action(temperature=temperature)
        
        # Record state and action
        trajectory.append((state.copy(), action, is_model1_turn))
        
        # Apply action and get new state
        state, reward, done = game.step(action)
        turn += 1
        
        if done:
            # Determine winner: reward is from perspective of player who just moved
            winner = 1 if is_model1_turn else 2
            return winner, trajectory
    
    # This should not happen in Nim
    return 0, trajectory


def run_tournament(boards, args):
    """Run a tournament between history and no-history models across multiple boards"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(lambda: defaultdict(Counter))
    
    for i, board in enumerate(boards):
        print(f"\nBoard {i+1}/{len(boards)}: {board}")
        board_tuple = tuple(board)  # Convert board to tuple for use as dictionary key
        
        # Load models
        _, history_model = load_model(
            board, 
            include_history=True, 
            alpha=args.alpha,
            checkpoint=args.checkpoint,
            device=device
        )
        
        game, no_history_model = load_model(
            board, 
            include_history=False, 
            alpha=args.alpha,
            checkpoint=args.checkpoint,
            device=device
        )
        
        # Play history first and no-history first games
        total_games = args.games
        history_first_games = total_games // 2
        no_history_first_games = total_games - history_first_games
        
        print(f"Playing {history_first_games} games with history model first...")
        for _ in tqdm(range(history_first_games)):
            winner, _ = play_game(
                game, 
                history_model, 
                no_history_model, 
                model1_first=True,
                num_simulations=args.simulations,
                temperature=args.temperature,
                exploration=args.exploration
            )
            if winner == 1:
                results[board_tuple]['history_first']['history_win'] += 1
            else:
                results[board_tuple]['history_first']['no_history_win'] += 1
        
        print(f"Playing {no_history_first_games} games with no-history model first...")
        for _ in tqdm(range(no_history_first_games)):
            winner, _ = play_game(
                game, 
                no_history_model, 
                history_model, 
                model1_first=True,
                num_simulations=args.simulations,
                temperature=args.temperature,
                exploration=args.exploration
            )
            if winner == 1:
                results[board_tuple]['no_history_first']['no_history_win'] += 1
            else:
                results[board_tuple]['no_history_first']['history_win'] += 1
    
    return results


def print_tournament_results(results):
    """Print formatted tournament results"""
    print("\n" + "="*80)
    print("TOURNAMENT RESULTS")
    print("="*80)
    
    # Calculate overall statistics
    overall_history_wins = 0
    overall_no_history_wins = 0
    
    # Board-by-board results
    for i, (board_tuple, scenarios) in enumerate(results.items()):
        board = list(board_tuple)  # Convert tuple back to list for display
        print(f"\nBoard {i+1}: {board}")
        board_history_wins = 0
        board_no_history_wins = 0
        
        # History first scenario
        history_first = scenarios['history_first']
        history_wins_as_first = history_first['history_win']
        no_history_wins_as_second = history_first['no_history_win']
        total_history_first = history_wins_as_first + no_history_wins_as_second
        
        print(f"  History first: History won {history_wins_as_first}/{total_history_first} ({history_wins_as_first/total_history_first*100:.1f}%)")
        
        # No-history first scenario
        no_history_first = scenarios['no_history_first']
        no_history_wins_as_first = no_history_first['no_history_win']
        history_wins_as_second = no_history_first['history_win']
        total_no_history_first = no_history_wins_as_first + history_wins_as_second
        
        print(f"  No-history first: No-history won {no_history_wins_as_first}/{total_no_history_first} ({no_history_wins_as_first/total_no_history_first*100:.1f}%)")
        
        # Combined board statistics
        board_history_wins = history_wins_as_first + history_wins_as_second
        board_no_history_wins = no_history_wins_as_first + no_history_wins_as_second
        board_total = board_history_wins + board_no_history_wins
        
        print(f"  Combined: History won {board_history_wins}/{board_total} ({board_history_wins/board_total*100:.1f}%)")
        
        # Add to overall counts
        overall_history_wins += board_history_wins
        overall_no_history_wins += board_no_history_wins
    
    # Overall statistics
    overall_total = overall_history_wins + overall_no_history_wins
    print("\n" + "-"*80)
    print("OVERALL RESULTS")
    print(f"History model wins: {overall_history_wins}/{overall_total} ({overall_history_wins/overall_total*100:.1f}%)")
    print(f"No-history model wins: {overall_no_history_wins}/{overall_total} ({overall_no_history_wins/overall_total*100:.1f}%)")
    
    # First/second player advantage analysis
    history_as_first = sum(scenarios['history_first']['history_win'] for scenarios in results.values())
    history_as_second = sum(scenarios['no_history_first']['history_win'] for scenarios in results.values())
    no_history_as_first = sum(scenarios['no_history_first']['no_history_win'] for scenarios in results.values())
    no_history_as_second = sum(scenarios['history_first']['no_history_win'] for scenarios in results.values())
    
    first_total = history_as_first + no_history_as_first
    second_total = history_as_second + no_history_as_second
    
    print("\nPLAYER ORDER ANALYSIS")
    print(f"First player wins: {first_total}/{first_total+second_total} ({first_total/(first_total+second_total)*100:.1f}%)")
    print(f"History as first player: {history_as_first}/{history_as_first+no_history_as_second} ({history_as_first/(history_as_first+no_history_as_second)*100:.1f}%)")
    print(f"History as second player: {history_as_second}/{history_as_second+no_history_as_first} ({history_as_second/(history_as_second+no_history_as_first)*100:.1f}%)")
    
    # Save results summary to file
    with open(f"tournament_results_alpha_{args.alpha}_temp_{args.temperature}_exp_{args.exploration}.txt", "w") as f:
        f.write("TOURNAMENT RESULTS\n")
        f.write(f"Configuration: Alpha={args.alpha}, Games={args.games}, Simulations={args.simulations}, "
                f"Temperature={args.temperature}, Exploration={args.exploration}\n\n")
        f.write(f"OVERALL: History {overall_history_wins}/{overall_total} ({overall_history_wins/overall_total*100:.1f}%) "
                f"vs No-history {overall_no_history_wins}/{overall_total} ({overall_no_history_wins/overall_total*100:.1f}%)\n\n")
        
        # Add first/second player statistics
        f.write("PLAYER ORDER ANALYSIS\n")
        f.write(f"First player advantage: {first_total/(first_total+second_total)*100:.1f}%\n")
        f.write(f"History as first player: {history_as_first/(history_as_first+no_history_as_second)*100:.1f}%\n")
        f.write(f"History as second player: {history_as_second/(history_as_second+no_history_as_first)*100:.1f}%\n\n")
        
        # Add per-board details
        f.write("PER-BOARD RESULTS\n")
        for i, (board_tuple, scenarios) in enumerate(results.items()):
            board = list(board_tuple)
            board_history_wins = scenarios['history_first']['history_win'] + scenarios['no_history_first']['history_win']
            board_total = sum(scenarios['history_first'].values()) + sum(scenarios['no_history_first'].values()) 
            f.write(f"Board {i+1} {board}: History {board_history_wins}/{board_total} "
                    f"({board_history_wins/board_total*100:.1f}%)\n")


def main():
    args = parse_args()
    
    # Define test boards (same as in generalization_analysis.py)
    # All have 25 total stones to ensure same action space
    all_boards = [
        # Simple positions
        [1, 3, 5, 7, 9],  # Training board
        [2, 4, 4, 6, 9],
        [3, 3, 4, 6, 9],
        
        # Medium positions
        [1, 1, 3, 8, 12],
        [5, 5, 5, 5, 5],
        [2, 2, 5, 7, 9],
        
        # Hard positions
        [1, 2, 3, 9, 10],
        [1, 1, 1, 8, 14],
        [3, 4, 6, 6, 6],
        
        # Extreme positions
        [1, 1, 1, 1, 21],
        [1, 1, 1, 2, 20],
        [1, 2, 2, 5, 15]
    ]
    
    # Filter boards based on user input
    if args.boards:
        try:
            board_indices = [int(idx) for idx in args.boards.split(',')]
            test_boards = [all_boards[idx] for idx in board_indices]
            print(f"Using {len(test_boards)} selected boards: {board_indices}")
        except (ValueError, IndexError) as e:
            print(f"Error parsing board indices: {e}")
            print(f"Using all {len(all_boards)} boards")
            test_boards = all_boards
    else:
        test_boards = all_boards
    
    # Print tournament configuration
    print(f"Tournament configuration:")
    print(f"- Alpha: {args.alpha}")
    print(f"- Games per board: {args.games}")
    print(f"- Simulations per move: {args.simulations}")
    print(f"- Temperature: {args.temperature}")
    print(f"- Exploration: {args.exploration}")
    print(f"- Checkpoint: {args.checkpoint if args.checkpoint else 'latest'}")
    
    # Run tournament
    results = run_tournament(test_boards, args)
    
    # Print results
    print_tournament_results(results)


if __name__ == "__main__":
    main()