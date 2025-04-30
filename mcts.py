import math
import numpy as np
import copy

# The default value for the alpha and epsilon is copied from self-play section of the paper,
# mastering the game of Go without human knowledge
def pucb_score(parent, child, c1=1.25, c2=19652):
    
    pb_c = c1 + math.log((parent.visit_count + c2 + 1) / c2)
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior

    if child.visit_count > 0:
        value_score = - child.value()
    else:
        value_score = 0
    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature=1):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action_probs = np.zeros_like(visit_counts)
            action_probs[np.argmax(visit_counts)] = 1.0
            action = np.random.choice(actions, p=action_probs)
        elif temperature == float('inf'):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        return action

    def select_child(self):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for i, (action, child) in enumerate(self.children.items()):
            score = pucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child
    
    def expand(self, state, to_play, actions, action_probs):
        self.to_play = to_play
        self.state = state
        for action, prob in zip(actions, action_probs):
            if prob != 0.0:
                self.children[action] = Node(prior=prob, to_play=self.to_play * -1)

    def add_dirichlet_noise(self, alpha, epsilon):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - epsilon) + n * epsilon


class MCTS:
    def __init__(self, game, model, args):
        self.game = copy.deepcopy(game)
        self.model = model
        self.args = args
    
    def run(self, state, to_play, is_train=True):
        root = Node(0, to_play)

        self.game.reset_board(state.copy())
        self.game.player = to_play
        
        action_probs, _ = self.model.predict(state)

        action_masks = self.game.action_masks()
        action_probs = action_probs * action_masks
        action_probs /= np.sum(action_probs)

        root.expand(state, to_play, self.game.all_legal_actions, action_probs)
        if is_train:
            root.add_dirichlet_noise(self.args['alpha'], epsilon=self.args['epsilon'])

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            self.game.reset_board(parent.state.copy())
            self.game.player = parent.to_play

            next_state, reward, done = self.game.step(action)

            if not done:
                action_probs_pred, value = self.model.predict(next_state)
                action_masks = self.game.action_masks()
                action_probs = action_probs_pred * action_masks
                action_probs /= np.sum(action_probs)
                node.expand(next_state, self.game.to_play(), self.game.all_legal_actions, action_probs)
            else:
                value = -reward

            self.backpropagate(search_path, value, self.game.to_play())
        
        return root

    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1


if __name__ == "__main__":
    from game import Nim
    from model import Nim_Model

    args = {
        'n_piles': 3,
        'num_simulations': 10000,
        'batch_size': 128,
        'numEps': 104,
        'numIters': 2000,
        'epochs': 1,
        'lr': 0.02,
        'milestones': [200, 600],
        'scheduler_gamma': 0.1,
        'weight_decay': 1e-4,
        'hidden_size': 16,
        'num_lstm_layers': 1,
        'num_head_layers': 1,
        'branching_factor': 1,
        'exploration_moves': 3,
        'num_samples': 10000,
        'alpha': 0.35,
        'epsilon': 0.25,
        'calculate_elo': False
    }

    init_board = [2 * i + 1 for i in range(args['n_piles'])]
    game = Nim(init_board=init_board, include_history=True, num_frames=2)
    model = Nim_Model(
        action_size=game.action_size,
        input_size=args['n_piles'],
        hidden_size=args['hidden_size'],
        num_lstm_layers=args['num_lstm_layers'],
        num_head_layers=args['num_head_layers']
    )

    root = MCTS(game, model, args).run(game.reset(), game.to_play())
    print([(game.unpack_action(action), child.visit_count) for action, child in root.children.items()])
