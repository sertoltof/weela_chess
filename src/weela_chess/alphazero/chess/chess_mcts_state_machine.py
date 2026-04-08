import numpy as np
from chess import Board, WHITE, BLACK
from numpy.typing import NDArray

from weela_chess.alphazero.aux_mcts_interfaces import MCTSStateMachine, StateEvaluator, StateEncoder


from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_categorical_idx, categorical_idx_to_uci_codes
from weela_chess.data_io.convert_pgn_to_np import board_to_matrix

UCI_TO_ACTION = all_uci_codes_to_categorical_idx()
ACTION_TO_UCI = categorical_idx_to_uci_codes()

# todo move these to another file
class ChessWinOrLossEvaluator(StateEvaluator):

    def get_global_state_value_from_others_perspective(self, global_state_value: float, other_state: 'ChessMctsStateMachine | None') -> float:
        """given state: state that is currently looking at the value."""
        # white always makes the first move
        if other_state is None:
            return global_state_value
        if other_state.board.turn == WHITE:
            return global_state_value
        return -global_state_value

    def state_value(self, state: 'ChessMctsStateMachine') -> float:
        # either the last player made the winning move, or it's still mid-game

        if state.check_is_over():
            # -1 (black) if it's white turn to move NOW (after the game is over)
            outcome = state.board.outcome(claim_draw=True)
            if outcome.winner is None:
                return 0
            return 1 if outcome.winner == WHITE else -1
        return 0


class MinimalChessStateEncoder(StateEncoder):

    @property
    def encoded_shape(self) -> tuple[int, ...]:
        return 17, 8, 8

    def encode_state(self, state: 'ChessMctsStateMachine') -> NDArray:
        # https://chatgpt.com/c/69b88ca2-f4a0-8329-9e99-69cce3a62f23
        # The full encoding is as follows:
        #  Each time step has:
        #   - 12 piece position planes
        #   - 1 plane for "this position has occured 2 or more times (including this time)
        #   - 1 plane for "this position has occured 3 or more times
        # there are up to 7 previous board states included as well (8 total)
        # 7 Global State Planes:
        #   - 4 planes for: current player can castle (white, black) x (queen side, king side)
        #   - 1 plane for the total moves (out of 100/200)
        #   - 1 plane for number of moves since the last pawn move, or capture
        #   - 1 plane for the side to move (white/black)

        # Minimal Encoding:
        #  board state/piece position (12 x 8 x 8)
        #  castle rights (4 x 8 x 8)
        #  side to move (1 x 8 x 8)

        full_encoding = np.zeros(shape=self.encoded_shape, dtype=np.float32)

        piece_encoding = board_to_matrix(state.board, from_whites_view=state.board.turn)
        # move the channel dimension (piece types) to the first dimension for pytorch
        piece_encoding = np.moveaxis(piece_encoding, source=-1, destination=0)
        full_encoding[0:12, :, :] = piece_encoding

        if state.board.has_queenside_castling_rights(color=WHITE):
            full_encoding[12] = np.ones(shape=(8, 8))
        if state.board.has_kingside_castling_rights(color=WHITE):
            full_encoding[13] = np.ones(shape=(8, 8))
        if state.board.has_queenside_castling_rights(color=BLACK):
            full_encoding[14] = np.ones(shape=(8, 8))
        if state.board.has_kingside_castling_rights(color=BLACK):
            full_encoding[15] = np.ones(shape=(8, 8))
        if state.board.turn == WHITE:
            full_encoding[16] = np.ones(shape=(8, 8))
        return full_encoding


class ChessMctsStateMachine(MCTSStateMachine):

    def __init__(self, board: Board = None, action_history: list[int] = None):
        if board is None:
            board = Board()
        if action_history is None:
            action_history = []
        self.board = board

        self.action_history: list[int] = action_history

        self.uci_to_action: dict[str, int] = UCI_TO_ACTION
        self.action_to_uci: dict[int, list[str]] = ACTION_TO_UCI

    @property
    def current_player(self) -> int:
        return 1 if self.board.turn == WHITE else -1

    @property
    def full_policy_size(self) -> int:
        return len(self.action_to_uci)

    @property
    def num_valid_actions(self) -> int:
        return len(self.valid_actions())

    def valid_actions(self) -> list[int] | NDArray:
        legal_moves = self.board.legal_moves
        valid_actions = [self.uci_to_action[x.uci()] for x in legal_moves]
        return valid_actions

    def valid_action_mask(self) -> list[int] | NDArray[int]:
        blank_mask = np.zeros(shape=(self.full_policy_size,), dtype=int)
        blank_mask[self.valid_actions()] = 1
        return blank_mask

    def take_action(self, action_idx: int) -> 'MCTSStateMachine':
        """Returns snapshot of the state after action is taken"""
        next_board, accepted_move = self.board.copy(), False
        for possible_uci in self.action_to_uci[action_idx]:
            try:
                next_board.push_uci(possible_uci)
                accepted_move = True
                break
            except ValueError as e:
                pass
        if not accepted_move:
            raise RuntimeError("Couldn't find a UCI that the board would accept")
        next_state_machine = ChessMctsStateMachine(next_board,
                                                   action_history=self.action_history + [action_idx])
        return next_state_machine

    def check_is_over(self) -> bool:
        if self.board.can_claim_draw():
            return True
        return self.board.is_game_over()
