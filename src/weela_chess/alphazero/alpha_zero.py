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


class AlphaZero:
    def __init__(self, model: Module, optimizer: Optimizer, root_state: MCTSStateMachine,
                 mcts: MCTS, train_params: AlphaZeroTrainParams,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.root_state = root_state

        self.mcts = mcts
        self.train_params = train_params
        self.device = device

    def next_move(self, game, state, embed_state) -> int:
        action_probs, current_value = self.model(embed_state)
        action_probs = action_probs.squeeze()
        action_probs = torch.softmax(action_probs, axis=0)
        valid_moves = game.get_valid_moves(state)
        action_probs *= valid_moves
        return int(np.argmax(action_probs))

    def self_play(self):
        memory: list[tuple[MCTSStateMachine, NDArray]] = []
        state_machine = self.root_state

        while True:
            action_probs = self.mcts.search(state_machine)
            memory.append((state_machine, action_probs))

            temperature_action_probs = action_probs ** (1 / self.train_params.temperature)
            temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            action = np.random.choice(state_machine.action_size, p=temperature_action_probs)  # change to p=temperature_action_probs

            state_machine = state_machine.take_action(action)
            value = state_machine.state_value()
            is_terminal = state_machine.check_is_over()

            if is_terminal:
                return_memory = []
                for hist_state, hist_action_probs in memory:
                    # hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    return_memory.append((
                        hist_state.get_encoded_state(),
                        hist_action_probs,
                        hist_state.get_my_value_from_parents_perspective(hist_state, value)
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

    def learn(self):
        for iteration in range(self.train_params.num_iterations):
            memory = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.train_params.num_self_play_before_train)):
                memory += self.self_play()

            self.model.train()
            for epoch in trange(self.train_params.num_train_epochs):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
