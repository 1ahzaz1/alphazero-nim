import torch
import numpy as np
import random
from game import Nim
from model import Nim_Model
from trainer import Trainer
import ray
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Nim AlphaZero model")
    parser.add_argument("--include_history", type=int, choices=[0, 1], default=1,
                        help="Include history in model (0=False, 1=True)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha parameter for Dirichlet noise")
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="Epsilon parameter for exploration")
    parser.add_argument("--iterations", type=int, default=300,
                        help="Number of training iterations")
    parser.add_argument("--simulations", type=int, default=100,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--self_play_games", type=int, default=200,
                        help="Number of self-play games per iteration")
    parser.add_argument("--min_temp", type=float, default=0.5,
                        help="Minimum temperature for move selection")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay for regularization")
    parser.add_argument("--learning_rate", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--random_starts", action="store_true",
                        help="Use random starting positions during training")
    parser.add_argument("--seed", type=int, default=30,
                        help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    ray.init(ignore_reinit_error=True)
    
    # Get training parameters from args
    include_history = bool(args.include_history)
    alpha = args.alpha
    epsilon = args.epsilon
    num_iterations = args.iterations
    num_simulations = args.simulations
    num_self_play_games = args.self_play_games
    min_temp = args.min_temp
    weight_decay = args.weight_decay
    learning_rate = args.learning_rate
    random_starts = args.random_starts
    
    # Number of workers for parallel self-play
    num_workers = 8  # Adjust based on your machine's capabilities
    
    # Base board configuration
    base_board = [1, 3, 5, 7, 9]  # WINNING POSITION (nim-sum = 1) with 5 piles
    
    # Alternative starting positions with same total stones (25)
    alternative_boards = [
        [1, 3, 5, 7, 9],   # Original
        [2, 4, 4, 6, 9],    # Variation 1
        [3, 3, 4, 6, 9],    # Variation 2
        [5, 5, 5, 5, 5],    # Symmetric
        [1, 1, 3, 8, 12]    # Asymmetric
    ]

    # Set up training parameters
    train_args = {
        'init_board': base_board,      
        'include_history': include_history,     
        'num_simulations': num_simulations,
        'batch_size': 128,
        'numEps': num_workers * (num_self_play_games // num_workers),  # Increased for more diverse data
        'numIters': num_iterations,
        'epochs': 4,
        'lr': learning_rate,
        'milestones': [100, 200],
        'scheduler_gamma': 0.2,
        'weight_decay': weight_decay,
        'hidden_size': 128,
        'num_lstm_layers': 1,
        'num_head_layers': 2,
        'alpha': alpha,
        'epsilon': epsilon,
        'min_temperature': min_temp,  # Minimum temperature for move selection
        'random_starts': random_starts,
        'alternative_boards': alternative_boards
    }

    # Folder name includes key parameters
    train_args['model_dir'] = f"./models/{len(base_board)}_{include_history}_{alpha}"
    
    # Display training configuration
    print("Training Configuration:")
    print(f"- History: {'Enabled' if include_history else 'Disabled'}")
    print(f"- Alpha: {alpha}")
    print(f"- Epsilon: {epsilon}")
    print(f"- Min Temperature: {min_temp}")
    print(f"- Self-play games per iteration: {train_args['numEps']}")
    print(f"- Training iterations: {num_iterations}")
    print(f"- Random starting positions: {'Enabled' if random_starts else 'Disabled'}")
    print(f"- Model will be saved to: {train_args['model_dir']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize game
    game = Nim(
        init_board=base_board,
        include_history=include_history,
        num_frames=2,
        alternative_boards=alternative_boards if random_starts else None
    )

    # Initialize model
    model = Nim_Model(
        action_size=game.action_size,
        input_size=len(base_board),
        hidden_size=train_args['hidden_size'],
        num_lstm_layers=train_args['num_lstm_layers'],
        num_head_layers=train_args['num_head_layers']
    )

    # Train model
    trainer = Trainer(game=game, model=model, args=train_args, device=device, num_workers=num_workers)
    trainer.learn()
