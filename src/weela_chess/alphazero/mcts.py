import math
from collections.abc import Callable

import loguru
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
import torch
import torch.nn as nn

from weela_chess.alphazero.aux_mcts_interfaces import MCTSStateMachine, StateEvaluator, StateEncoder


class MCTSConfig(BaseModel):
    c: float

    dirichlet_epsilon: float
    dirichlet_alpha: float

    @property
    def C(self) -> float:
        return self.c


class Node:
    def __init__(self, config: MCTSConfig, state_machine: MCTSStateMachine, state_evaluator: StateEvaluator,
                 parent: "Node | None" = None, action_taken: int | None = None,
                 prior: float = 0, visit_count: int = 0):
        self.config = config
        self.state_machine = state_machine
        self.state_evaluator = state_evaluator

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
        """todo make sure this is the same as
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
                child = Node(self.config, child_state, self.state_evaluator, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value: float):
        # The value here is the state value of this state from the global perspective
        parents_state = self.parent.state_machine if self.parent is not None else None
        my_value = self.state_evaluator.get_global_state_value_from_others_perspective(value, parents_state)
        # we record the value from the parent's perspective here, because they are the one looking at this node to decide whether or not to enter it
        self.value_sum += my_value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(my_value)


MCTSModel = Callable[[MCTSStateMachine], tuple[list[float] | NDArray, float | NDArray]]
"""Given a state, return the action policy, value tensors"""


def build_torch_mcts_model(torch_model: nn.Module, state_encoder: StateEncoder, device: torch.device) -> MCTSModel:
    def call_model(state_machine: MCTSStateMachine) -> tuple[list[float] | NDArray, float | NDArray]:
        policy, value = torch_model(
            torch.tensor(state_encoder.encode_state(state_machine), device=device).unsqueeze(0)
        )
        # since the torch model always returns the relative value (from the perspective of the moving player), we need to convert back to absolute before returning
        value = value.item()
        if state_machine.current_player == -1:
            value *= -1

        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        return policy, value

    return call_model


class MCTS:
    def __init__(self, config: MCTSConfig, full_action_size: int, model: MCTSModel,
                 terminal_state_evaluator: StateEvaluator):
        self.config = config
        self.state_machine_action_size = full_action_size
        self.model = model
        self.terminal_state_evaluator = terminal_state_evaluator

    @staticmethod
    def filter_valid_actions(state: MCTSStateMachine, policy: list[float] | NDArray) -> list[float] | NDArray:
        valid_moves = state.valid_action_mask()
        policy *= valid_moves
        policy /= np.sum(policy)
        return policy

    @torch.no_grad()
    def get_policy_and_val_preds(self, state: MCTSStateMachine, with_dirichlet: bool = False) -> tuple[list[float], float]:
        policy, value = self.model(state)
        if with_dirichlet:
            policy = (1 - self.config.dirichlet_epsilon) * policy
            policy += self.config.dirichlet_epsilon * np.random.dirichlet([self.config.dirichlet_alpha] * self.state_machine_action_size)
        policy = self.filter_valid_actions(state, policy)
        return policy, value

    @torch.no_grad()
    def search(self, state: MCTSStateMachine, num_searches: int,
               is_eval: bool = False):
        root = Node(self.config, state, self.terminal_state_evaluator, visit_count=1)

        policy, _ = self.get_policy_and_val_preds(state, with_dirichlet=True if not is_eval else False)
        root.expand(policy)

        for search in range(num_searches):
            loguru.logger.debug(f"Starting Search #{search}/{num_searches}")
            node = root

            while node.is_fully_expanded():
                node = node.select()

            is_terminal = node.state_machine.check_is_over()

            if not is_terminal:
                policy, value = self.get_policy_and_val_preds(node.state_machine)
                # make sure the model is trained so that it predicts of the value of the player that moved into the given state
                node.expand(policy)
            else:
                value = self.terminal_state_evaluator.state_value(node.state_machine)
            # value here is the player that moved INTO the state
            node.backpropagate(value)

        action_probs = np.zeros(self.state_machine_action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs = self.filter_valid_actions(state, action_probs)
        return action_probs
