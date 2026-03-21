import numpy as np
from numpy.typing import NDArray

from weela_chess.alphazero.tic_tac_toe.tic_tac_toe import TicTacToe
from weela_chess.alphazero.mcts import MCTSStateMachine

from torch.types import Tensor


class TTTMctsStateMachine(MCTSStateMachine):

    def __init__(self, game: TicTacToe, state: NDArray | None = None, whose_turn: int = 1):
        self.game = game
        self.state = state
        self.whose_turn: int = whose_turn
        """the player whose turn it is, looking at the given state"""

        self.action_history: list[int] = []

        """1 for player 1, -1 for player 2"""
        if state is None:
            self.state = game.get_initial_state()

    @property
    def action_size(self) -> int:
        return len(self.valid_actions())

    def valid_actions(self) -> list[int] | NDArray:
        return np.argwhere((self.state.reshape(-1) == 0)).flatten().astype(np.uint8)

    def valid_action_mask(self) -> list[int] | NDArray[int]:
        return (self.state.reshape(-1) == 0).astype(np.uint8)

    def take_action(self, action_idx: int) -> 'MCTSStateMachine':
        """Returns snapshot of the state after action is taken"""
        next_state = self.game.get_next_state(self.state, action_idx, self.whose_turn)
        next_state_machine = TTTMctsStateMachine(self.game, next_state, self.whose_turn * -1)
        next_state_machine.action_history.append(action_idx)
        return next_state_machine

    # def peek_action(self, action_idx: int) -> 'MCTSStateMachine':
    #     """Returns snapshot of the state after action is taken. No internal state change"""
    #     pass

    def get_my_value_from_parents_perspective(self, parents_state: 'TTTMctsStateMachine | None', value: float) -> float:
        """given state: state that is currently looking at the value."""
        if parents_state is None:
            return -value
        if parents_state.whose_turn == self.whose_turn:
            return value
        return -value

    def state_value(self) -> float:
        value, is_over = self.game.get_value_and_terminated(self.state, self.action_history[-1])
        # player two made the previous move
        if self.whose_turn == 1:
            value *= -1
        return value

    def naive_predicted_state_value(self) -> float:
        # # you care about the state value from the perspective of the player that just made the move, not the player whose move it is
        if self.check_is_over():
            return self.state_value()
        desired_direction = -self.whose_turn
        state = self.state * desired_direction
        row_sums = np.max(np.sum(state, axis=0))
        col_sums = np.max(np.sum(state, axis=1))
        diag_a_sums = np.max(np.diag(state))
        diag_b_sums = np.max(np.diag(np.flip(state, axis=0)))
        biggest_sum = np.max([row_sums, col_sums, diag_a_sums, diag_b_sums])
        return biggest_sum / max([self.game.row_count, self.game.column_count])

        # num_one_aways = 0
        # for valid_action in experiment_state.valid_actions():
        #     if_move_was_made = experiment_state.take_action(valid_action)
        #     if if_move_was_made.check_is_over():
        #         num_one_aways += 1
        # return num_one_aways / (self.game.row_count * self.game.column_count)

    def check_is_over(self) -> bool:
        value, is_over = self.game.get_value_and_terminated(self.state, self.action_history[-1])
        return is_over

    def get_encoded_state(self) -> Tensor:
        if self.whose_turn == 1:
            return self.game.get_encoded_state(self.state)
        else:
            return self.game.get_encoded_state(-self.state)

    def copy(self) -> 'TTTMctsStateMachine':
        return TTTMctsStateMachine(self.game, self.state.copy(), self.whose_turn)
