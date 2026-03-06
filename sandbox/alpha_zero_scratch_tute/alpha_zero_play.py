import torch

import numpy as np
from sandbox.alpha_zero_scratch_tute.alpha_zero_train import AlphaZero
from sandbox.alpha_zero_scratch_tute.tic_tac_toe import TicTacToe
from sandbox.alpha_zero_scratch_tute.ttt_policy_net import TTTResNet

if __name__ == '__main__':
    device = torch.device("cpu")

    tictactoe = TicTacToe()
    model = TTTResNet(tictactoe, 4, 64, device)
    model.load_state_dict(torch.load("results/model_2.pt"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(torch.load("results/optimizer_2.pt"))

    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }
    alphaZero = AlphaZero(model, optimizer, tictactoe, args)
    torch.set_grad_enabled(False)

    state = tictactoe.get_initial_state()
    player = 1
    while True:
        print(state)

        if player == 1:
            valid_moves = tictactoe.get_valid_moves(state)
            print("valid_moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue
        else:
            neutral_state = tictactoe.change_perspective(state, player)
            embed_state = tictactoe.get_encoded_state(state)
            torch_state = torch.tensor(embed_state, dtype=torch.float32).unsqueeze(0)
            action = alphaZero.next_move(tictactoe, neutral_state, torch_state)

        state = tictactoe.get_next_state(state, action, player)

        value, is_terminal = tictactoe.get_value_and_terminated(state, action)

        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = tictactoe.get_opponent(player)
