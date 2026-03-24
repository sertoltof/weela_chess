import numpy as np
from chess import Board
from numpy.typing import NDArray

from weela_chess.alphazero.tic_tac_toe.tic_tac_toe import TicTacToe
from weela_chess.alphazero.mcts import MCTSStateMachine

from torch.types import Tensor

from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_categorical_idx, categorical_idx_to_uci_codes
from weela_chess.data_io.convert_pgn_to_np import board_to_matrix


class ChessMctsStateMachine(MCTSStateMachine):

    def __init__(self, board: Board, state: NDArray | None = None, whose_turn: int = 1,
                 action_history: list[int] = None):
        if action_history is None:
            action_history = []
        self.board = board
        self.state = state
        self.whose_turn: int = whose_turn
        """the player whose turn it is, looking at the given state
        1 is white, -1 is black."""

        self.action_history: list[int] = action_history

        if state is None:
            self.state = board_to_matrix(self.board, from_whites_view=whose_turn == 1)
        self.uci_to_action: dict[str, int] = all_uci_codes_to_categorical_idx()
        self.action_to_uci: dict[int, list[str]] = categorical_idx_to_uci_codes()

    @property
    def action_size(self) -> int:
        return len(self.valid_actions())

    def valid_actions(self) -> list[int] | NDArray:
        legal_moves = self.board.legal_moves
        valid_actions = [self.uci_to_action[x.uci()] for x in legal_moves]
        return valid_actions

    def valid_action_mask(self) -> list[int] | NDArray[int]:
        blank_mask = np.zeros(shape=(self.action_size,), dtype=int)
        blank_mask[self.valid_actions()] = 1
        return blank_mask

    def take_action(self, action_idx: int) -> 'MCTSStateMachine':
        """Returns snapshot of the state after action is taken"""
        next_board, accepted_move = self.board.copy(), False
        for possible_uci in self.action_to_uci[action_idx]:
            try:
                next_board.push_uci(possible_uci)
                break
            except ValueError as e:
                pass
        if not accepted_move:
            raise RuntimeError("Couldn't find a UCI that the board would accept")
        next_state_machine = ChessMctsStateMachine(next_board, whose_turn=self.whose_turn * -1,
                                                   action_history=self.action_history + [action_idx])
        return next_state_machine

    def get_my_value_from_parents_perspective(self, parents_state: 'ChessMctsStateMachine | None', value: float) -> float:
        """given state: state that is currently looking at the value."""
        if parents_state is None:
            return -value
        if parents_state.whose_turn == self.whose_turn:
            return value
        return -value

    def state_value(self) -> float:
        # either the last player made the winning move, or it's still mid-game
        if self.check_is_over():
            return self.whose_turn * -1
        return 0

    def naive_predicted_state_value(self) -> float:
        # todo implement some simple naive state, such as the amount of material each side has
        return self.state_value()

    def check_is_over(self) -> bool:
        return self.board.is_game_over()

    def get_encoded_state(self) -> Tensor:
        return self.state