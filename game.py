import numpy as np


class Nim:
    def __init__(self, init_board=[2, 2, 2, 2, 2], include_history=True, num_frames=2, alternative_boards=None):
        super(Nim, self).__init__()
        self.init_board = init_board
        self.n_piles = len(init_board)
        self.board = init_board.copy()
        self.alternative_boards = alternative_boards

        self.include_history = include_history
        self.num_frames = num_frames
        self.history = []

        self.all_legal_actions = self.legal_actions()
        self.all_legal_actions_idx = {action: idx for idx, action in enumerate(self.all_legal_actions)}
        self.action_size = len(self.all_legal_actions)

        self.player = 1

    def reset_board(self, state):
        state = state.copy()
        if self.include_history:
            total_len = self.n_piles * (self.num_frames + 1)
            flat = state[:total_len]
            unflattened = [flat[i:i + self.n_piles] for i in range(0, len(flat), self.n_piles)]
            self.history = unflattened[:-1]
            self.board = unflattened[-1]
        else:
            self.board = state

    def to_play(self):
        return self.player

    def reset(self, board=None):
        # Use provided board, or default to init_board
        if board is not None:
            if len(board) != self.n_piles:
                raise ValueError(f"Board must have {self.n_piles} piles, got {len(board)}")
            self.board = board.copy()
        else:
            self.board = self.init_board.copy()
            
        if self.include_history:
            self.history = []
            
        self.player = 1  # Reset player to 1
        return self.state()

    def state(self):
        if self.include_history:
            padded_history = [[0] * self.n_piles for _ in range(self.num_frames - len(self.history))] + self.history
            full_sequence = padded_history + [self.board]
            return [val for frame in full_sequence for val in frame]
        else:
            return self.board.copy()

    def step(self, action):
        pile_idx, take = self.unpack_action(action)
        before_take = self.board[pile_idx]
        after_take = before_take - take

        if self.include_history:
            self.history.append(self.board.copy())
            if len(self.history) > self.num_frames:
                self.history.pop(0)

        self.board[pile_idx] = after_take

        done = sum(self.board) == 0
        reward = 1.0 if done else 0.0
        state = self.state()

        self.player *= -1
        return state.copy(), reward, done

    def legal_actions(self):
        actions = []
        for (pile_idx, take) in enumerate(self.board):
            if take > 0:
                for i in range(take):
                    action = i * self.n_piles + pile_idx
                    actions.append(action)
        return actions.copy()

    def action_masks(self):
        mask = [0.0 for _ in range(len(self.all_legal_actions))]
        legal_actions = self.legal_actions()
        for i, action in enumerate(self.all_legal_actions):
            if action in legal_actions:
                mask[i] = 1.0
        return mask.copy()

    def unpack_action(self, action):
        pile_idx = action % self.n_piles
        take = int((action - pile_idx) / self.n_piles + 1)
        return pile_idx, take

    def winning_position(self, board):
        xor = 0
        for c in board:
            xor = c ^ xor
        return xor == 0

    def winning_move(self):
        actions = self.legal_actions()
        for action in actions:
            board = self.board.copy()
            pile_idx, take = self.unpack_action(action)
            board[pile_idx] = self.board[pile_idx] - take
            if self.winning_position(board):
                return action
        return np.random.choice(actions)


# Debug block for standalone testing
if __name__ == "__main__":
    print("Multi-frame Nim Debug Mode\n")

    # Testing different configurations
    game = Nim(init_board=[1, 3, 5], include_history=True, num_frames=2)

    state = game.reset()
    print(f"Initial state (flattened): {state}\n")

    done = False
    step = 0
    while not done:
        action = game.winning_move()
        print(f"Step {step}: Taking action {game.unpack_action(action)}")

        state, reward, done = game.step(action)
        print(f"New state (flattened): {state}")
        print(f"Internal board: {game.board}")
        print(f"History (most recent last): {game.history}\n")

        step += 1

    print(f"Game ended with reward: {reward}")
