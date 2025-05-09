import os
import re
import numpy as np
import torch
from game import Nim
from mcts import MCTS
from model import Nim_Model
from main import set_seed

# --- CONFIG (edit this part only) ---
include_history = True
board = [1, 3, 5, 7, 9]  # WINNING POSITION (nim-sum = 1) with 5 piles
num_frames = 2
num_simulation = 100  # REDUCED: Use fewer simulations to highlight network differences
model_base_dir = './models'
training_alpha = 0.8  # Alpha value used during TRAINING (for directory path)
analysis_alpha = 0.8  # Alpha value to use during ANALYSIS (for exploration)
temperature = 1.0  # Higher temperature to see more move diversity
# -------------------------------------

set_seed(30)

def win_lose_position(board):
    xor = 0
    for c in board:
        xor = c ^ xor
    # From the perspective of the player who just made the move:
    return 'GOOD MOVE (opponent in losing position)' if xor == 0 else 'BAD MOVE (opponent in winning position)'

# Get the model folder name - use training_alpha for the directory
model_folder = f"{len(board)}_{include_history}_{training_alpha}"
model_dir = os.path.join(model_base_dir, model_folder)

# Find latest checkpoint
checkpoints = [f for f in os.listdir(model_dir) if re.match(r'checkpoint_iter_\d+\.pt', f)]
latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
model_path = os.path.join(model_dir, latest_checkpoint)

print(f"Using model: {model_path}")
print("History enabled" if include_history else "History disabled")

# Init game/model - IMPORTANT: Must match the training architecture EXACTLY
game = Nim(board, include_history, num_frames=num_frames)
model = Nim_Model(action_size=game.action_size,
                  input_size=5,  # Must match number of heaps (5)
                  hidden_size=128,
                  num_lstm_layers=1,
                  num_head_layers=2)  # MUST match training (now 2 layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))

# Run MCTS on board state with limited search to emphasize network quality
mcts = MCTS(game, model, {
    'num_simulations': num_simulation, 
    'alpha': analysis_alpha,
    'epsilon': 0.5  # Add exploration but not too much
})
state = game.state()

# Use training mode to include some exploration
root = mcts.run(state, game.to_play(), is_train=True)

# Display root stats
_, value = model.predict(root.state)
print(f'\n🟨 root: {board} ({win_lose_position(board)}) | V: {round(value, 2)} | WL: {round((0.5 + root.value()/2) * 100, 2)}%')

# Print children and select action with temperature
total_visits = sum(child.visit_count for child in root.children.values())
sorted_children = sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True)

# Get action probabilities based on temperature
visit_counts = np.array([child.visit_count for child in root.children.values()])
actions = list(root.children.keys())
if temperature == 0:
    action_probs = np.zeros_like(visit_counts)
    action_probs[np.argmax(visit_counts)] = 1.0
else:
    visit_count_distribution = visit_counts ** (1 / temperature)
    action_probs = visit_count_distribution / sum(visit_count_distribution)

print(f"\nTop moves (using temperature={temperature}):")
for action, child in sorted_children:
    if child.state is not None:
        child_board = child.state[len(board)*num_frames:] if include_history else child.state
        predicted_value = model.predict(child.state)[1]
        print(f"{game.unpack_action(action)}: {child_board} ({win_lose_position(child_board)}) | "
              f"P: {round(child.prior, 2)} | "
              f"V: {round(-predicted_value, 2)} | "
              f"N: {child.visit_count} ({round(child.visit_count/total_visits * 100, 2)}%) | "
              f"Q: {round(-child.value(), 2)} | "
              f"WL: {round(0.5 - child.value()/2, 2)}")
