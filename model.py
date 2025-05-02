import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os


class Nim_Model(nn.Module):
    def __init__(self, action_size, input_size=10, hidden_size=128, num_lstm_layers=1, num_head_layers=1):
        super(Nim_Model, self).__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # input_size = number of heaps per frame (i.e. frame width)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.policy_head = nn.Sequential(
            *[nn.Linear(hidden_size, hidden_size) for _ in range(num_head_layers - 1)],
            nn.Linear(hidden_size, self.action_size)
        )
        self.value_head = nn.Sequential(
            *[nn.Linear(hidden_size, hidden_size) for _ in range(num_head_layers - 1)],
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        batch_size, total_seq_len = x.shape
        if len(x.shape) != 2:
            raise ValueError("Input must be of shape (batch_size, sequence_length)")

        # Infer n_frames and n_piles
        n_piles = self.lstm.input_size
        if total_seq_len % n_piles != 0:
            raise ValueError("Input sequence length is not divisible by number of heaps (n_piles)")

        n_frames = total_seq_len // n_piles
        x = x.view(batch_size, n_frames, n_piles)

        self.lstm.flatten_parameters()
        h0 = x.new_zeros(self.num_lstm_layers, batch_size, self.hidden_size)
        c0 = x.new_zeros(self.num_lstm_layers, batch_size, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]  # use the final LSTM output
        action_logits = self.policy_head(out)
        value_logit = self.value_head(out)

        return F.softmax(action_logits, dim=-1), torch.tanh(value_logit)

    def predict(self, state):
        state = np.array(state)
        if len(state.shape) != 1:
            raise Exception('predict() only supports a single input (1D vector).')

        device = next(self.parameters()).device
        state = torch.FloatTensor(state.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            policy, value = self(state)

        return policy.squeeze().cpu().numpy(), value.item()

    def save_checkpoint(self, folder='.', filename='checkpoint_model'):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder, filename))

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


# 🧪 Debug/test block
if __name__ == '__main__':
    from game import Nim

    print("Testing Nim_Model with multi-frame input")

    # Setup game with history
    game = Nim(init_board=[1, 3, 5], include_history=True, num_frames=2)
    game.reset()
    game.step(game.winning_move())  # build history
    game.step(game.winning_move())

    # Get multi-frame input
    state = game.state()
    print(f"\nState from game.py (flattened): {state}")

    # Prepare model
    n_piles = len(game.init_board)
    model = Nim_Model(action_size=game.action_size, input_size=n_piles).to('cpu')

    # Run through model
    state_tensor = torch.FloatTensor([state])
    policy, value = model(state_tensor)

    print("\nModel output:")
    print("Policy:", policy.detach().numpy())
    print("Value:", value.item())
