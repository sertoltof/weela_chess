import numpy as np
from numpy.typing import NDArray

from weela_chess.alphazero.tic_tac_toe.tic_tac_toe import TicTacToe
from weela_chess.alphazero.aux_mcts_interfaces import MCTSStateMachine, StateEvaluator, StateEncoder

from torch.types import Tensor


class TTTStateEvaluator(StateEvaluator):

    def get_global_state_value_from_others_perspective(self, global_state_value: float, other_state: 'TTTMctsStateMachine | None') -> float:
        """This function is basically just to flip the value as you backprop up the node tree"""
        if other_state is None:
            return global_state_value
        if other_state.whose_turn == 1:
            return global_state_value
        return -global_state_value

    def state_value(self, state: 'TTTMctsStateMachine') -> float:
        """Return the value of the state from the global perspective"""
        value, is_over = state.game.get_value_and_terminated(state.state, state.action_history[-1])
        # player two made the previous (winning) move
        if state.whose_turn == 1:
            value *= -1
        return value


class TTTStateEncoder(StateEncoder):

    def __init__(self, game: TicTacToe):
        self.game = game

    @property
    def encoded_shape(self) -> tuple[int, ...]:
        return 3, self.game.row_count, self.game.column_count

    def encode_state(self, state: 'TTTMctsStateMachine') -> NDArray:
        if state.whose_turn == 1:
            return state.game.get_encoded_state(state.state)
        else:
            return state.game.get_encoded_state(-state.state)


class TTTMctsStateMachine(MCTSStateMachine):

    def __init__(self, game: TicTacToe, state: NDArray | None = None, whose_turn: int = 1, action_history: list[int] = None):
        self.game = game
        self.state = state
        self.whose_turn: int = whose_turn
        """the player whose turn it is, looking at the given state"""
        if action_history is None:
            action_history = []
        self.action_history: list[int] = action_history

        """1 for player 1, -1 for player 2"""
        if state is None:
            self.state = game.get_initial_state()

    @property
    def current_player(self) -> int:
        return self.whose_turn

    @property
    def full_policy_size(self) -> int:
        return self.game.column_count * self.game.row_count

    @property
    def num_valid_actions(self) -> int:
        return len(self.valid_actions())

    def valid_actions(self) -> list[int] | NDArray:
        return np.argwhere((self.state.reshape(-1) == 0)).flatten().astype(np.uint8)

    def valid_action_mask(self) -> list[int] | NDArray[int]:
        return (self.state.reshape(-1) == 0).astype(np.uint8)

    def take_action(self, action_idx: int) -> 'MCTSStateMachine':
        """Returns snapshot of the state after action is taken"""
        next_state = self.game.get_next_state(self.state, action_idx, self.whose_turn)
        next_state_machine = TTTMctsStateMachine(self.game, next_state, self.whose_turn * -1, action_history=list(self.action_history))
        next_state_machine.action_history.append(action_idx)
        return next_state_machine

    def check_is_over(self) -> bool:
        value, is_over = self.game.get_value_and_terminated(self.state, self.action_history[-1])
        return is_over

    def copy(self) -> 'TTTMctsStateMachine':
        return TTTMctsStateMachine(self.game, self.state.copy(), self.whose_turn)

    # def naive_predicted_state_value(self) -> float:
    #     # # you care about the state value from the perspective of the player that just made the move, not the player whose move it is
    #     if self.check_is_over():
    #         return self.state_value()
    #     desired_direction = -self.whose_turn
    #     state = self.state * desired_direction
    #     row_sums = np.max(np.sum(state, axis=0))
    #     col_sums = np.max(np.sum(state, axis=1))
    #     diag_a_sums = np.max(np.diag(state))
    #     diag_b_sums = np.max(np.diag(np.flip(state, axis=0)))
    #     biggest_sum = np.max([row_sums, col_sums, diag_a_sums, diag_b_sums])
    #     return biggest_sum / max([self.game.row_count, self.game.column_count])
