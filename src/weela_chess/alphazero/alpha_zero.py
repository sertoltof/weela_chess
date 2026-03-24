import pickle
from typing import Callable

from numpy._typing import NDArray
from pydantic import BaseModel
from torch.nn import Module
from torch.optim import Optimizer
import torch
import numpy as np
from tqdm import tqdm, trange

from weela_chess.alphazero.mcts import MCTSModel, MCTSStateMachine, MCTS
import torch.nn.functional as F
import random


class AlphaZeroTrainParams(BaseModel):
    temperature: float
    batch_size: int
    """number of moves to batch train at a time"""
    num_iterations: int
    """number of times to perform self-play + train loop"""
    num_self_play_before_train: int

    num_train_epochs: int
    """how much to repeat-train the same training memory in deep model"""


class AlphaZeroTrainer:
    def __init__(self, model: Module, optimizer: Optimizer, root_state_factory: Callable[[], MCTSStateMachine],
                 mcts: MCTS, train_params: AlphaZeroTrainParams,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.root_state_factory = root_state_factory

        self.mcts = mcts
        self.train_params = train_params
        self.device = device

    def self_play(self, num_searches: int, with_priors: bool = True) -> list[tuple[MCTSStateMachine, list[float] | NDArray, float]]:
        memory: list[tuple[MCTSStateMachine, NDArray]] = []
        root_state = self.root_state_factory()
        state_machine = root_state

        while True:
            action_probs = self.mcts.search(state_machine, num_searches=num_searches,
                                            no_priors=not with_priors)
            memory.append((state_machine, action_probs))

            temperature_action_probs = action_probs ** (1 / self.train_params.temperature)
            temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            action = np.random.choice(root_state.action_size, p=temperature_action_probs)  # change to p=temperature_action_probs

            state_machine = state_machine.take_action(action)
            is_terminal = state_machine.check_is_over()

            if is_terminal:
                value = state_machine.state_value()

                # value of the player that made the winning move (not the player in the current state)
                return_memory: list[tuple[MCTSStateMachine, list[float] | NDArray, float]] = []
                for hist_state, hist_action_probs in memory:
                    # value is absolute now, one player won, that's what that position should be predicted to lead towards
                    return_memory.append((
                        hist_state,
                        hist_action_probs,
                        value
                    ))
                return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.train_params.batch_size):
            sample = memory[batchIdx:min(len(memory) - 1,
                                         batchIdx + self.train_params.batch_size)]  # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()  # change to self.optimizer
            loss.backward()
            self.optimizer.step()  # change to self.optimizer

    def learn(self, num_iterations: int, num_self_play_per_iter: int,
              num_train_epochs_per_iter: int, num_mcts_searches: int,
              with_priors: bool = True, first_iteration_num: int = 0):
        iteration = first_iteration_num
        for new_iteration in range(num_iterations):
            full_memory: list[tuple[MCTSStateMachine, list[float], float]] = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(num_self_play_per_iter)):
                full_memory += self.self_play(with_priors=with_priors, num_searches=num_mcts_searches)
                # if selfPlay_iteration % 5 == 0:
                #     memory += self.rando_dummy_play(with_priors=with_priors)

            train_memory = [(x[0].get_encoded_state(), x[1], x[2]) for x in full_memory]
            self.model.train()
            for epoch in trange(num_train_epochs_per_iter):
                self.train(train_memory)

            torch.save(self.model.state_dict(), f"model_{iteration + new_iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration + new_iteration}.pt")
            with open(f"memory_{iteration + new_iteration}.pkl", "wb") as f:
                pickle.dump(full_memory, f)

    def snapshot_training_state(self, version: int):
        torch.save(self.model.state_dict(), f"model_{version}.pt")
        torch.save(self.optimizer.state_dict(), f"optimizer_{version}.pt")


class AlphaZeroPlayer:
    def __init__(self, model: Module, mcts: MCTS, device: torch.device):
        self.model = model
        # self.root_state_factory = root_state_factory

        self.mcts = mcts
        self.device = device

    def make_next_move_with_search(self, from_state: MCTSStateMachine, num_searches: int) -> int:
        action_probs = self.mcts.search(from_state, num_searches=num_searches,
                                        is_eval=True)
        action = int(np.argmax(action_probs))
        return action

    def make_next_move_model_only(self, from_state: MCTSStateMachine):
        action_probs, _ = self.mcts.get_policy_and_val_preds(from_state)
        action = int(np.argmax(action_probs))
        return action
