import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from torch.types import Tensor
from pydantic import BaseModel
import torch


class MCTSStateMachine(ABC):

    def __init__(self, start_state: NDArray):
        self.state = start_state.copy()

    @abstractmethod
    @property
    def action_size(self) -> int:
        pass

    @abstractmethod
    def valid_actions(self, state: NDArray) -> list[int] | NDArray:
        pass

    @abstractmethod
    def take_action(self, action_idx: int) -> 'MCTSStateMachine':
        """Returns snapshot of the state after action is taken"""
        pass

    @abstractmethod
    def peek_action(self, action_idx: int) -> 'MCTSStateMachine':
        """Returns snapshot of the state after action is taken. No internal state change"""
        pass

    @abstractmethod
    def get_value_given_state(self, given_state: 'MCTSStateMachine', value: float, base_state: 'MCTSStateMachine') -> float:
        """given state: state that is currently looking at the value.
        base_state: state that originally generated the value"""
        pass

    @abstractmethod
    def state_value(self) -> float:
        pass

    @abstractmethod
    def get_encoded_state(self) -> Tensor:
        pass


class MCTSConfig(BaseModel):
    c: float
    num_searches: int

    dirichlet_epsilon: float
    dirichlet_alpha: float

    @property
    def C(self) -> float:
        return self.c


class Node:
    def __init__(self, config: MCTSConfig, state_machine: MCTSStateMachine,
                 parent: "Node | None" = None, action_taken: int | None = None,
                 prior: float = 0, visit_count: int = 0):
        self.config = config
        self.state_machine = state_machine
        # self.state = state

        self.parent = parent
        self.action_taken = action_taken
        """action taken by the parent to get to this node"""
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0
        """sum of this node value and value of all descendents"""

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        """todo this is supposed to factor in the mean value of the next state (Q + U)
            https://lczero.org/dev/lc0/search/alphazero/"""
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: 'Node'):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.config.C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy: list[float]):
        for action, prob in enumerate(policy):
            if prob > 0.0:
                # child_state = self.state.copy()
                child_state = self.state_machine.take_action(action)

                child = Node(self.config, self.state_machine, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value: float):
        value = self.state_machine.get_value_for_state(self.state, value)
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)


MCTSModel = Callable[[Tensor], tuple[list[float] | Tensor, float | Tensor]]
"""Given a state, return the action policy, value tensors"""


class MCTS:
    def __init__(self, config: MCTSConfig, state_machine: MCTSStateMachine, model: MCTSModel, device: torch.device):
        self.config = config
        self.state_machine = state_machine
        self.model = model
        self.device = device

    @torch.no_grad()
    def search(self, state: NDArray):
        root = Node(self.config, self.state_machine, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(self.state_machine.get_encoded_state(state), device=self.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.config.dirichlet_epsilon) * policy
        policy += self.config.dirichlet_epsilon * np.random.dirichlet([self.config.dirichlet_alpha] * self.state_machine.action_size)

        valid_moves = self.state_machine.valid_actions(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.config.num_searches):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
