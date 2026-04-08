import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Iterable

import numpy as np
from numpy._typing import NDArray
from numpy.typing import NDArray
from torch.types import Tensor
from pydantic import BaseModel
import torch
import torch.nn as nn


class MCTSStateMachine(ABC):

    @property
    @abstractmethod
    def current_player(self) -> int:
        """convention is 1 for player 1, -1 for player 2. I'm only working on 1/2 player games, so that's all I need"""
        pass

    @property
    @abstractmethod
    def full_policy_size(self) -> int:
        pass

    @property
    @abstractmethod
    def num_valid_actions(self) -> int:
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

    @abstractmethod
    def check_is_over(self) -> bool:
        pass


class StateEvaluator(ABC):

    @abstractmethod
    def get_global_state_value_from_others_perspective(self, global_state_value: float, other_state: 'MCTSStateMachine | None') -> float:
        """This function is basically just to flip the value as you backprop up the node tree"""
        pass

    @abstractmethod
    def state_value(self, state: MCTSStateMachine) -> float:
        """Return the value of the state from the global perspective"""
        pass


class StateEncoder(ABC):

    @property
    @abstractmethod
    def encoded_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def encode_state(self, state: MCTSStateMachine) -> NDArray:
        pass
