import numpy as np
from random import shuffle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from mcts import MCTS
import ray
import copy
import os

@ray.remote
class Simulation:
    def __init__(self, game, args):
        self.game = copy.deepcopy(game)
        self.args = args

    def execute_episode(self, model):
        self.model = model
        mcts = MCTS(self.game, self.model, self.args)
        train_examples = []
        
        state = self.game.reset()
        done = False
        n_moves = 0
        with torch.no_grad():
            while not done:
                root = mcts.run(state, self.game.to_play(), is_train=True)
                action_probs = [0.0 for _ in range(self.game.action_size)]
                for action, child in root.children.items():
                    action_probs[self.game.all_legal_actions_idx[action]] = child.visit_count
                action_probs = action_probs / np.sum(action_probs)

                train_examples.append((state, action_probs, self.game.to_play()))

                # Temperature scheduling
                if n_moves <= 3:
                    temp = 2.0
                elif 3 < n_moves < 5:
                    temp = 1.0
                else:
                    temp = 0

                action = root.select_action(temperature=temp)
                next_state, reward, done = self.game.step(action)
                state = next_state

                n_moves += 1

                if done:
                    examples = []
                    for history_state, history_action_probs, history_player in train_examples:
                        examples.append((history_state, history_action_probs,
                                         -reward if history_player == self.game.to_play() else reward))
                    return examples


class Trainer:
    def __init__(self, game, model, args, device, num_workers=4):
        self.game = game
        self.model = model
        self.args = args
        self.device = device
        self.batch_counter = 0
        self.epoch_counter = 0

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args['lr'],
                                    weight_decay=args['weight_decay'])
        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=args['milestones'],
                                     gamma=args['scheduler_gamma'])

        self.num_workers = num_workers
        self.simulations = [Simulation.remote(self.game, self.args) for _ in range(self.num_workers)]

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):
            print(f'{i}/{self.args["numIters"]}')
            train_examples = []
            self.model.to(torch.device('cpu'))
            self.model.eval()
            for _ in range(self.args['numEps'] // self.num_workers):
                examples = ray.get([sim.execute_episode.remote(self.model) for sim in self.simulations])
                for exp in examples:
                    train_examples.extend(exp)
            shuffle(train_examples)
            self.train(train_examples)
            if i % 50 == 0 or i == self.args['numIters']:
                if not os.path.exists(self.args['model_dir']):
                    os.makedirs(self.args['model_dir'])
                self.model.save_checkpoint(folder=self.args['model_dir'], filename=f"checkpoint_iter_{i}.pt")

    def train(self, examples):
        for _ in range(self.args['epochs']):
            batch_idx = 0
            while batch_idx < len(examples) // self.args['batch_size']:
                self.model.to(self.device)
                self.model.train()

                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.FloatTensor(np.array(boards)).contiguous().to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).contiguous().to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).contiguous().to(self.device)

                out_pi, out_v = self.model(boards)
                loss_pi = self.loss_pi(target_pis, out_pi)
                loss_v = self.loss_v(target_vs, out_v)
                total_loss = loss_pi + loss_v
                print(f"Iter {self.epoch_counter+1}, Batch {batch_idx+1} | Loss π: {loss_pi.item():.4f} | Loss V: {loss_v.item():.4f} | Total: {total_loss.item():.4f}")

                


                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                batch_idx += 1
                self.batch_counter += 1

            self.epoch_counter += 1
            self.scheduler.step()

    def loss_pi(self, targets, outputs):
        loss = - (targets * torch.log(outputs + 1e-8)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.mean((targets - outputs.squeeze()) ** 2)
        return loss


# ✅ Debug block to test training with dummy multi-frame inputs
if __name__ == "__main__":
    from game import Nim
    from model import Nim_Model

    print("🧪 Running debug mode for Trainer...")

    args = {
        'n_piles': 3,
        'batch_size': 4,
        'epochs': 1,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'milestones': [1],
        'scheduler_gamma': 0.1,
        'num_lstm_layers': 1,
        'num_head_layers': 1,
        'hidden_size': 16
    }

    game = Nim(init_board=[1, 3, 5], include_history=True, num_frames=2)
    model = Nim_Model(
        action_size=game.action_size,
        input_size=len(game.init_board),
        hidden_size=args['hidden_size'],
        num_lstm_layers=args['num_lstm_layers'],
        num_head_layers=args['num_head_layers']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(game, model, args, device)

    # Create dummy training data
    examples = []
    for _ in range(10):
        state = game.reset()
        game.step(game.winning_move())
        game.step(game.winning_move())
        state = game.state()
        action_probs = np.ones(game.action_size) / game.action_size
        value = np.random.uniform(-1, 1)
        examples.append((state, action_probs, value))

    trainer.train(examples)
    print("✅ Trainer ran successfully with dummy data.")
