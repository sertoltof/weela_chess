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

    @property
    @abstractmethod
    def full_policy_size(self) -> int:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        pass

    @abstractmethod
    def valid_actions(self) -> list[int] | NDArray[int]:
        pass

    @abstractmethod
    def valid_action_mask(self) -> list[int] | NDArray[int]:
        pass

    @abstractmethod
    def take_action(self, action_idx: int) -> 'MCTSStateMachine':
        """Returns snapshot of the state after action is taken"""
        pass

    # @abstractmethod
    # def peek_action(self, action_idx: int) -> 'MCTSStateMachine':
    #     """Returns snapshot of the state after action is taken. No internal state change"""
    #     pass

    @abstractmethod
    def get_my_value_from_parents_perspective(self, parent: 'MCTSStateMachine | None', value: float) -> float:
        """base_state: state that originally declared the value"""
        pass

    @abstractmethod
    def state_value(self) -> float:
        """Return the value of the state from the perspective of the player that moved INTO the state"""
        pass

    @abstractmethod
    def naive_predicted_state_value(self) -> float:
        pass

    @abstractmethod
    def check_is_over(self) -> bool:
        pass

    @abstractmethod
    def get_encoded_state(self) -> Tensor:
        pass


class MCTSConfig(BaseModel):
    c: float

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
            q_value = child.value_sum / child.visit_count
        return q_value + self.config.C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy: list[float]):
        for action, prob in enumerate(policy):
            if prob > 0.0:
                child_state = self.state_machine.take_action(action)
                child = Node(self.config, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value: float):
        # the value for the nodes is always the opposite of the state value, because when you look at the nodes, you're always looking
        # from the perspective of the parent, i.e. the person that made the previous move.
        parents_state = self.parent.state_machine if self.parent is not None else None
        self.value_sum += value
        self.visit_count += 1

        value = self.state_machine.get_my_value_from_parents_perspective(parents_state, value)
        if self.parent is not None:
            self.parent.backpropagate(value)


MCTSModel = Callable[[Tensor], tuple[list[float] | Tensor, float | Tensor]]
"""Given a state, return the action policy, value tensors"""


class MCTS:
    def __init__(self, config: MCTSConfig, full_action_size: int, model: MCTSModel, device: torch.device):
        self.config = config
        self.state_machine_action_size = full_action_size
        self.model = model
        self.device = device

    @staticmethod
    def filter_valid_actions(state: MCTSStateMachine, policy: list[float] | NDArray) -> list[float] | NDArray:
        valid_moves = state.valid_action_mask()
        policy *= valid_moves
        policy /= np.sum(policy)
        return policy

    @torch.no_grad()
    def get_policy_and_val_preds(self, state: MCTSStateMachine, with_dirichlet: bool = False) -> tuple[list[float], float]:
        policy, value = self.model(
            torch.tensor(state.get_encoded_state(), device=self.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        if with_dirichlet:
            policy = (1 - self.config.dirichlet_epsilon) * policy
            policy += self.config.dirichlet_epsilon * np.random.dirichlet([self.config.dirichlet_alpha] * self.state_machine_action_size)
        policy = self.filter_valid_actions(state, policy)

        return policy, value.item()

    @torch.no_grad()
    def search(self, state: MCTSStateMachine, num_searches: int,
               no_priors: bool = False, is_eval: bool = False):
        root = Node(self.config, state, visit_count=1)

        if no_priors:
            policy = [0.3 + (0.4 * np.random.random()) for _ in range(self.state_machine_action_size)]
            policy = self.filter_valid_actions(state, policy)
        else:
            policy, _ = self.get_policy_and_val_preds(state, with_dirichlet=True if not is_eval else False)
        root.expand(policy)

        for search in range(num_searches):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            if no_priors:
                value = node.state_machine.naive_predicted_state_value()
            else:
                value = node.state_machine.state_value()
                # value here is the player that moved INTO the state
            is_terminal = node.state_machine.check_is_over()

            if not is_terminal:
                if no_priors:
                    policy = [0.3 + (0.4 * np.random.random()) for _ in range(self.state_machine_action_size)]
                    policy = self.filter_valid_actions(node.state_machine, policy)
                else:
                    policy, value = self.get_policy_and_val_preds(node.state_machine)
                    # make sure the model is trained so that it predicts of the value of the player that moved into the given state
                node.expand(policy)
            # else:
            #     print("hit the end when searching")

            node.backpropagate(value)

        action_probs = np.zeros(self.state_machine_action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs = self.filter_valid_actions(state, action_probs)
        return action_probs
