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
board = [2 for _ in range(10)]
num_frames = 2
num_simulation = 10000
model_base_dir = './models'
alpha = 0.35
# -------------------------------------

set_seed(30)

def win_lose_position(board):
    xor = 0
    for c in board:
        xor = c ^ xor
    return 'OPPONENT WIN (bad move)' if xor == 0 else ' OPPONENT LOSE (good move)'

# Get the model folder name
model_folder = f"{len(board)}_{include_history}_{alpha}"
model_dir = os.path.join(model_base_dir, model_folder)

# Find latest checkpoint
checkpoints = [f for f in os.listdir(model_dir) if re.match(r'checkpoint_iter_\d+\.pt', f)]
latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
model_path = os.path.join(model_dir, latest_checkpoint)

print(f"Using model: {model_path}")
print("History enabled" if include_history else "History disabled")

# Init game/model - IMPORTANT: Use hidden_size=64 which matches the saved model
game = Nim(board, include_history, num_frames=num_frames)
model = Nim_Model(action_size=game.action_size,
                  input_size=10,     # Explicitly set input_size to 10
                  hidden_size=128,   # Change to 128 to match newly trained model
                  num_lstm_layers=1,
                  num_head_layers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))

# Run MCTS on board state
mcts = MCTS(game, model, {'num_simulations': num_simulation, 'alpha': alpha})
state = game.state()
root = mcts.run(state, game.to_play(), is_train=False)

# Display root stats
_, value = model.predict(root.state)
print(f'\n🟨 root: {board} ({win_lose_position(board)}) | V: {round(value, 2)} | WL: {round((0.5 + root.value()/2) * 100, 2)}%')

# Print children
total_visits = sum(child.visit_count for child in root.children.values())
sorted_children = sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True)

print("\nTop moves:")
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
