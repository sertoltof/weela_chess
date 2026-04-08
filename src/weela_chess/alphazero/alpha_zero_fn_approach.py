from itertools import chain
import pickle
from typing import Callable
import loguru

from numpy._typing import NDArray
from pydantic import BaseModel
from torch.nn import Module
from torch.optim import Optimizer
import torch
import numpy as np
from tqdm import tqdm, trange

from weela_chess.alphazero.mcts import MCTSModel, MCTS
from weela_chess.alphazero.aux_mcts_interfaces import MCTSStateMachine, StateEvaluator, StateEncoder
import torch.nn.functional as F
import random
from pydantic import ConfigDict

from weela_chess.models.np_basemodel_support import NumpyArray


class TurnState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_num: int
    state: MCTSStateMachine
    policy: NumpyArray
    # value is from the perspective of the player that moved into the `state`
    value: float


def sample_temperature_probs(raw_probs: list[float] | NDArray[np.float32], temp: float) -> int:
    temperature_action_probs = raw_probs ** (1 / temp)
    temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs)
    action = np.random.choice(len(raw_probs), p=temperature_action_probs)  # change to p=temperature_action_probs
    return action


def self_mcts_play(first_state: MCTSStateMachine, num_searches: int,
                   mcts: MCTS, mcts_temperature: float,
                   final_state_evaluator: StateEvaluator,
                   max_moves: int | None = None, early_terminate_evaluator: StateEvaluator | None = None) -> list[TurnState]:
    memory: list[tuple[MCTSStateMachine, NDArray, int]] = []
    root_state = first_state
    state_machine = root_state
    move_num = 0

    while True:
        loguru.logger.debug(f">>>>Starting Turn #{move_num + 1}")

        action_probs = mcts.search(state_machine, num_searches=num_searches)
        action = sample_temperature_probs(action_probs, mcts_temperature)
        memory.append((state_machine, action_probs, move_num))

        state_machine = state_machine.take_action(action)
        move_num += 1
        if max_moves is not None and move_num > max_moves:
            value = early_terminate_evaluator.state_value(state_machine)
            loguru.logger.debug(f"Game terminated early, final value: {value}")
            return capture_run_memory(memory, value, final_state_evaluator)

        is_terminal = state_machine.check_is_over()
        if is_terminal:
            value = final_state_evaluator.state_value(state_machine)
            loguru.logger.debug(f"Game completed, final value: {value}")
            return capture_run_memory(memory, value, final_state_evaluator)


def two_player_mcts_play(first_state: MCTSStateMachine, num_searches: int,
                         mcts: MCTS, mcts_player: int, mcts_temperature: float,
                         final_state_evaluator: StateEvaluator,
                         player_two: Callable[[MCTSStateMachine], int]) -> list[TurnState]:
    memory: list[tuple[MCTSStateMachine, NDArray, int]] = []
    root_state = first_state
    state_machine = root_state
    move_num = 0

    while True:
        if state_machine.current_player == mcts_player:
            action_probs = mcts.search(state_machine, num_searches=num_searches)
            action = sample_temperature_probs(action_probs, mcts_temperature)
            memory.append((state_machine, action_probs, move_num))

            state_machine = state_machine.take_action(action)
        else:
            p2_action = player_two(state_machine)
            state_machine = state_machine.take_action(p2_action)
        move_num += 1

        is_terminal = state_machine.check_is_over()
        if is_terminal:
            value = final_state_evaluator.state_value(state_machine)
            return capture_run_memory(memory, value, final_state_evaluator)


def capture_run_memory(memory: list[tuple[MCTSStateMachine, list[float] | NDArray[np.float32], int]],
                       final_value: float,
                       state_evaluator: StateEvaluator) -> list[TurnState]:
    return_memory: list[TurnState] = []
    for hist_state, hist_action_probs, move_num in memory:
        value_from_states_perspective = state_evaluator.get_global_state_value_from_others_perspective(final_value, hist_state)
        return_memory.append(TurnState(
            turn_num=move_num,
            state=hist_state,
            policy=hist_action_probs,
            value=value_from_states_perspective
        ))
    return return_memory


def train_torch_model_on_results(model: Module, optimizer: Optimizer,
                                 num_train_epochs: int, game_results: list[list[TurnState]], state_encoder: StateEncoder,
                                 batch_size: int, device: torch.device):
    # full_memory: list[tuple[MCTSStateMachine, list[float], float]] = []
    flat_turns: list[TurnState] = list(chain.from_iterable(game_results))

    memory = [(state_encoder.encode_state(x.state), x.policy, x.value) for x in flat_turns]
    model.train()
    for epoch in trange(num_train_epochs):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), batch_size):
            sample = memory[batchIdx:min(len(memory) - 1,
                                         batchIdx + batch_size)]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=device)

            out_policy, out_value = model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            optimizer.zero_grad()  # change to self.optimizer
            loss.backward()
            optimizer.step()  # change to self.optimizer
