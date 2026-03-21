import torch

import numpy as np
from tqdm import tqdm

from sandbox.alpha_zero_scratch_tute.alpha_zero_train import AlphaZero
from src.weela_chess.alphazero.tic_tac_toe.tic_tac_toe import TicTacToe
from sandbox.alpha_zero_scratch_tute.ttt_policy_net import TTTResNet

from src.weela_chess.alphazero.tic_tac_toe.tic_tac_toe import TicTacToe
from sandbox.alpha_zero_scratch_tute.ttt_policy_net import TTTResNet
from weela_chess.alphazero.alpha_zero import AlphaZeroTrainer, AlphaZeroTrainParams, AlphaZeroPlayer
from weela_chess.alphazero.mcts import MCTSStateMachine, MCTS, MCTSConfig
from weela_chess.alphazero.tic_tac_toe.ttt_mcts_state_machine import TTTMctsStateMachine

N_GAMES = 20

if __name__ == '__main__':
    tictactoe = TicTacToe(row_count=5, column_count=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TTTResNet(tictactoe, 4, 50, device)
    model.load_state_dict(torch.load("./model_0.pt"))

    mcts_config = MCTSConfig(
        c=2.5,
        num_searches=60,

        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.4
    )
    root_state = TTTMctsStateMachine(tictactoe)
    state = root_state
    state.whose_turn = np.random.choice([-1, 1])
    mcts = MCTS(mcts_config, state, model, device)

    # alphaZero = AlphaZeroTrainer(model, optimizer, root_state,
    #                       mcts, train_params, device)
    # state.whose_turn = 1

    alphaZero = AlphaZeroPlayer(model, state, mcts, device)
    az_wins, az_losses = 0, 0
    game_results = []
    for i in tqdm(range(N_GAMES)):
        state = root_state
        while True:
            if state.whose_turn == 1:
                az_action = alphaZero.make_next_move_with_search(state, num_searches=100)
                # az_action = alphaZero.make_next_move_model_only(state)
                state = state.take_action(az_action)
                if state.check_is_over():
                    if state.state_value() == 1:
                        az_wins += 1
                        game_results.append(1)
                    else:
                        game_results.append(0)
                    break
            else:
                random_policy = [1.0] * root_state.action_size
                random_policy = MCTS.filter_valid_actions(state, random_policy)
                random_action = np.random.choice(root_state.action_size, p=random_policy)

                state = state.take_action(random_action)
                if state.check_is_over():
                    if state.state_value() == 1:
                        az_losses += 1
                        game_results.append(-1)
                    else:
                        game_results.append(0)
                    break

    print(f"Alpha Zero won {az_wins}/{N_GAMES} games")
    print(f"Alpha Zero lost {az_losses}/{N_GAMES} games")
    print(f"There were {N_GAMES - az_wins - az_losses}/{N_GAMES} draws")

    print(f"loss idxs: {np.argwhere(np.array(game_results) == -1).flatten()}")
