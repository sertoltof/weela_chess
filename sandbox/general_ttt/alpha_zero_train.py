import cProfile
import io
import pstats
from typing import Callable, Any

import numpy as np
import sys
from tqdm import tqdm

from src.weela_chess.alphazero.tic_tac_toe.tic_tac_toe import TicTacToe
from sandbox.alpha_zero_scratch_tute.ttt_policy_net import TTTResNet
from weela_chess.alphazero.alpha_zero import AlphaZeroTrainer, AlphaZeroTrainParams
from weela_chess.alphazero.mcts import MCTSStateMachine, MCTS, MCTSConfig
from weela_chess.alphazero.tic_tac_toe.ttt_mcts_state_machine import TTTMctsStateMachine

print(np.__version__)

import torch

print(torch.__version__)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())
print()

import torch.nn.functional as F

torch.manual_seed(0)

# def profile_func_cumulative_time(func: Callable[[], Any]) -> str:
#     """Profile a function and all of the functions that were called within it using cprofile"""
#     pr = cProfile.Profile()
#     pr.enable()
#     func()
#     pr.disable()
#     stream = io.StringIO()
#     stats = pstats.Stats(pr, stream=stream).sort_stats("cumtime")
#     stats.print_stats()
#     return stream.getvalue()

def build_ttt_root_state() -> TTTMctsStateMachine:
    root_state = TTTMctsStateMachine(tictactoe)
    root_state.whose_turn = np.random.choice([-1, 1])
    return root_state

if __name__ == '__main__':
    tictactoe = TicTacToe(row_count=5, column_count=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTTResNet(tictactoe, 4, 50, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    mcts_config = MCTSConfig(
        c=2.5,
        num_searches=60,

        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3
    )
    root_state_factory = build_ttt_root_state
    mcts = MCTS(mcts_config, build_ttt_root_state().action_size, model, device)

    train_params = AlphaZeroTrainParams(
        temperature=1.25,
        batch_size=64,
        num_iterations=8,
        num_self_play_before_train=10,
        num_train_epochs=4
    )
    alphaZero = AlphaZeroTrainer(model, optimizer, root_state_factory,
                                 mcts, train_params, device)

    # def profile_naive():
        # train the model initially with only MCTS searches
    alphaZero.learn(num_iterations=3, num_self_play_per_iter=300, num_train_epochs_per_iter=4,
                    with_priors=False, num_mcts_searches=100)
    # print(profile_func_cumulative_time(profile_naive))
    # sys.exit(0)

    # then, train it using the model as the priors
    alphaZero.learn(num_iterations=10, num_self_play_per_iter=100, num_train_epochs_per_iter=4,
                    with_priors=True, first_iteration_num=3, num_mcts_searches=60)