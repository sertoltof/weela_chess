# Callable[[MCTSStateMachine], int]
from typing import Callable
from chess.engine import UciProtocol, Cp, Mate
from numpy.typing import NDArray

from weela_chess.alphazero.aux_mcts_interfaces import StateEvaluator, MCTSStateMachine
from weela_chess.alphazero.chess.chess_mcts_state_machine import ChessMctsStateMachine, ChessWinOrLossEvaluator
import asyncio
import chess
import numpy as np
from chess import Move


async def next_stockfish_move(state: ChessMctsStateMachine, limit: chess.engine.Limit) -> int:
    transport, engine = await chess.engine.popen_uci(
        "/home/garrickw/rl_learning/weela_chess/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")
    uci = await engine.play(state.board, limit).move.uci()
    return state.uci_to_action[uci]


def stockfish_player_fn(engine: chess.engine.SimpleEngine, limit_sec: float) -> Callable[[ChessMctsStateMachine], int]:
    limit = chess.engine.Limit(limit_sec)

    def next_move(state: ChessMctsStateMachine) -> int:
        uci = engine.play(state.board, limit).move.uci()
        return state.uci_to_action[uci]

    return next_move


# MCTSModel = Callable[[MCTSStateMachine], tuple[list[float] | NDArray, float | NDArray]]
def stockfish_eval_fn(engine: chess.engine.SimpleEngine, limit_sec: float,
                      uniform_policy: bool) -> Callable[[ChessMctsStateMachine], tuple[list[float] | NDArray, float | NDArray]]:
    """Always returns the absolute evaluation (from white's perspective)"""
    limit = chess.engine.Limit(limit_sec)

    def eval_state(state: ChessMctsStateMachine) -> tuple[list[float] | NDArray, float | NDArray]:
        state_info = engine.analyse(state.board, limit)
        policy = None
        if uniform_policy:
            policy = np.array([1.0] * len(state.action_to_uci))
        else:
            # could read stockfish's move recommendations somehow, but I have no need for this currently
            pass
        score = state_info["score"].relative
        if isinstance(score, Cp):
            # 15 pawns ahead is basically max
            max_cp_score = 100 * 15
            # still reserve 0.1 of the score for when you have force mate on the board
            score = (int(score.cp) / max_cp_score) * 0.9
        elif isinstance(score, Mate):
            min_mate_score = 10
            moves_till_mate_score = min_mate_score - score.moves
            moves_till_mate_score = max(1, moves_till_mate_score) / min_mate_score
            score = 0.9 + (moves_till_mate_score * 0.1)
        if state.board.turn == chess.BLACK:
            score = -score
        return policy, -score

    return eval_state


class StockfishEvaluator(StateEvaluator):

    def __init__(self, engine: chess.engine.SimpleEngine, limit_sec: float,
                 uniform_policy: bool):
        self.engine = engine
        self.limit = chess.engine.Limit(limit_sec)
        self.uniform_policy = uniform_policy
        self.win_loss_evaler = ChessWinOrLossEvaluator()
        self.eval_fn = stockfish_eval_fn(engine, limit_sec, uniform_policy)

    def get_global_state_value_from_others_perspective(self, global_state_value: float, other_state: 'ChessMctsStateMachine | None') -> float:
        """given state: state that is currently looking at the value."""
        # white always makes the first move
        if other_state is None:
            return global_state_value
        if other_state.board.turn == chess.WHITE:
            return global_state_value
        return -global_state_value

    def state_value(self, state: ChessMctsStateMachine) -> float:
        if state.check_is_over():
            return self.win_loss_evaler.state_value(state)
        return self.eval_fn(state)[1]


if __name__ == '__main__':
    engine = chess.engine.SimpleEngine.popen_uci("/home/garrickw/rl_learning/weela_chess/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")
    try:
        stockfish_player = stockfish_player_fn(engine, limit_sec=0.3)
        stockfish_evaler = stockfish_eval_fn(engine, limit_sec=0.3, uniform_policy=True)

        state = ChessMctsStateMachine(None, None)

        while True:
            next_move = stockfish_player(state)
            print(f"took move: {state.action_to_uci[next_move]}")
            state_eval = stockfish_evaler(state)

            state = state.take_action(next_move)
            if state.check_is_over():
                break
    finally:
        engine.quit()
