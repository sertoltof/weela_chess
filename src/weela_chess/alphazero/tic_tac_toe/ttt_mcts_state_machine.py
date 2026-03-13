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
        """1 for player 1, -1 for player 2"""
        if state is None:
            self.state = game.get_initial_state()

    @property
    def action_size(self) -> int:
        return int(np.count_nonzero(self.valid_actions()))

    def valid_actions(self) -> list[int] | NDArray:
        return (self.state.reshape(-1) == 0).astype(np.uint8)

    def take_action(self, action_idx: int) -> 'MCTSStateMachine':
        """Returns snapshot of the state after action is taken"""
        # todo finish implementing. I've decided that everything from the perspective of the state machine stays in the realm of
        #  1s are player 1s moves, -1s are player 2s moves. But, in the encoded state, we need to prepare it in the way the model wants it to be,
        #  that is the next move maker is 1 and -1 is opponent. Not sure how chess is going to work, maybe the turn is baked into the state somehow.
        next_state = self.game.get_next_state(self.state, action_idx, 1)



    # def peek_action(self, action_idx: int) -> 'MCTSStateMachine':
    #     """Returns snapshot of the state after action is taken. No internal state change"""
    #     pass

    def get_value_given_state(self, given_state: 'MCTSStateMachine', value: float) -> float:
        """given state: state that is currently looking at the value."""
        pass

    def state_value(self) -> float:
        pass

    def get_encoded_state(self) -> Tensor:
        pass
