import cProfile
import io
import pstats
from typing import Callable, Any

import loguru
import numpy as np
import sys
from tqdm import tqdm

from src.weela_chess.alphazero.tic_tac_toe.tic_tac_toe import TicTacToe
from sandbox.alpha_zero_scratch_tute.ttt_policy_net import TTTResNet
from weela_chess.alphazero.alpha_zero import AlphaZeroTrainer, AlphaZeroTrainParams
from weela_chess.alphazero.alpha_zero_fn_approach import self_mcts_play, train_torch_model_on_results
from weela_chess.alphazero.mcts import MCTS, MCTSConfig, build_torch_mcts_model
from weela_chess.alphazero.aux_mcts_interfaces import MCTSStateMachine
from weela_chess.alphazero.tic_tac_toe.ttt_mcts_state_machine import TTTMctsStateMachine, TTTStateEvaluator, TTTStateEncoder

print(np.__version__)

import torch

print(torch.__version__)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())
print()

import torch.nn.functional as F

torch.manual_seed(0)

def profile_func_cumulative_time(func: Callable[[], Any]) -> str:
    """Profile a function and all of the functions that were called within it using cprofile"""
    pr = cProfile.Profile()
    pr.enable()
    func()
    pr.disable()
    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream).sort_stats("cumtime")
    stats.print_stats()
    return stream.getvalue()

def build_ttt_root_state() -> TTTMctsStateMachine:
    root_state = TTTMctsStateMachine(tictactoe)
    root_state.whose_turn = np.random.choice([-1, 1])
    return root_state

if __name__ == '__main__':
    tictactoe = TicTacToe(row_count=5, column_count=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTTResNet(tictactoe, 4, 32, device)
    print(f"Model has: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    mcts_config = MCTSConfig(
        c=2.5,

        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3
    )
    state_evaluator = TTTStateEvaluator()
    state_encoder = TTTStateEncoder(tictactoe)

    root_state_factory = build_ttt_root_state
    mcts_model = build_torch_mcts_model(model, state_encoder, device)
    mcts = MCTS(mcts_config, build_ttt_root_state().full_policy_size, mcts_model,
                terminal_state_evaluator=state_evaluator)

    train_params = AlphaZeroTrainParams(
        temperature=1.25,
        batch_size=64,
        num_iterations=8,
        num_self_play_before_train=10,
        num_train_epochs=4
    )
    loguru.logger.remove()
    loguru.logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    for i in range(10):
        games = []

        model.eval()
        for g in tqdm(range(20)):
            # def profile():
            game = self_mcts_play(TTTMctsStateMachine(tictactoe), num_searches=60,
                                  mcts=mcts, mcts_temperature=1.25,
                                  final_state_evaluator=state_evaluator)
            games.append(game)
            # print(profile_func_cumulative_time(profile))
            # sys.exit(1)

        train_torch_model_on_results(model, optimizer,
                                     num_train_epochs=4, game_results=games, state_encoder=state_encoder,
                                     batch_size=64, device=device)
        torch.save(model.state_dict(), f"model_{i}.pt")
        torch.save(optimizer.state_dict(), f"optimizer_{i}.pt")


    # alphaZero = AlphaZeroTrainer(model, optimizer, root_state_factory,
    #                              mcts, train_params, device)
    #
    # # def profile_naive():
    #     # train the model initially with only MCTS searches
    # alphaZero.learn(num_iterations=3, num_self_play_per_iter=300, num_train_epochs_per_iter=4,
    #                 with_priors=False, num_mcts_searches=100)
    # # print(profile_func_cumulative_time(profile_naive))
    # # sys.exit(0)
    #
    # # then, train it using the model as the priors
    # alphaZero.learn(num_iterations=10, num_self_play_per_iter=100, num_train_epochs_per_iter=4,
    #                 with_priors=True, first_iteration_num=3, num_mcts_searches=60)