import numpy as np
from tqdm import tqdm

from src.weela_chess.alphazero.tic_tac_toe.tic_tac_toe import TicTacToe
from sandbox.alpha_zero_scratch_tute.ttt_policy_net import TTTResNet
from weela_chess.alphazero.alpha_zero import AlphaZero, AlphaZeroTrainParams
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

if __name__ == '__main__':
    tictactoe = TicTacToe()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTTResNet(tictactoe, 4, 64, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # args = {
    #     'C': 2,
    #     'num_searches': 60,
    #     'num_iterations': 3,
    #     'num_selfPlay_iterations': 500,
    #     'num_epochs': 4,
    #     'batch_size': 64,
    #     'temperature': 1.25,
    #     'dirichlet_epsilon': 0.25,
    #     'dirichlet_alpha': 0.3
    # }

    mcts_config = MCTSConfig(
        c=2,
        num_searches=60,

        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3
    )
    root_state = TTTMctsStateMachine(tictactoe)
    mcts = MCTS(mcts_config, root_state, model, device)

    train_params = AlphaZeroTrainParams(
        temperature=1.25,
        batch_size=64,
        num_iterations=3,
        num_self_play_before_train=50,
        num_train_epochs=4
    )
    alphaZero = AlphaZero(model, optimizer, root_state,
                          mcts, train_params, device)
    alphaZero.learn()
