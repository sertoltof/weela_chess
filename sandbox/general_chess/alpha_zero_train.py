import numpy as np

from weela_chess.alphazero.chess.chess_res_net import ChessResNet
from weela_chess.alphazero.alpha_zero import AlphaZeroTrainer, AlphaZeroTrainParams
from weela_chess.alphazero.chess.chess_mcts_state_machine import ChessMctsStateMachine
from weela_chess.alphazero.mcts import MCTS, MCTSConfig

print(np.__version__)

import torch

print(torch.__version__)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())
print()

torch.manual_seed(0)


def build_chess_root_state() -> ChessMctsStateMachine:
    root_state = ChessMctsStateMachine()
    return root_state


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessResNet(5, 50, device,
                        n_input_channels=17)
    print(f"Model has: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    mcts_config = MCTSConfig(
        c=2.5,

        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3
    )
    root_state_factory = build_chess_root_state
    mcts = MCTS(mcts_config, len(build_chess_root_state().action_to_uci), model, device)

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
    # alphaZero.learn(num_iterations=3, num_self_play_per_iter=300, num_train_epochs_per_iter=4,
    #                 with_priors=False, num_mcts_searches=100)
    # print(profile_func_cumulative_time(profile_naive))
    # sys.exit(0)

    # then, train it using the model as the priors
    alphaZero.learn(num_iterations=10, num_self_play_per_iter=2, num_train_epochs_per_iter=4,
                    with_priors=True, first_iteration_num=0, num_mcts_searches=60)
