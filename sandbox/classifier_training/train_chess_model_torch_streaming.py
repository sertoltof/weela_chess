import asyncio
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from chess import Board
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np

from torch_utils.torch_train_utils import pytorch_train, pytorch_test
from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves, categorical_idx_to_uci_codes
from weela_chess.chess_utils.stockfish_eval import PLAYER_FUNC, play_against_stockfish
from weela_chess.data_io.pgn_db_utils import pgn_np_move_streamer, board_to_matrix
from weela_chess.datasets.torch_move_dataset_factory import TorchMoveDataLoaderFactory


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(8, 64, 3, 1)
        # self.conv2 = nn.Conv2d(64, 128, 3, 1)
        # self.conv3 = nn.Conv2d(128, 128, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.0),
            nn.Dropout(0.0),
            nn.Dropout(0.0),
            nn.Dropout(0.15)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(768, 768),
            nn.Linear(768, 768),
            nn.Linear(768, 256),
            nn.Linear(256, len(categorical_idx_to_uci_codes()))
        ])

        # self.fc1 = nn.Linear(1536, 1536)
        # self.fc2 = nn.Linear(1536, 256)
        # self.fc3 = nn.Linear(256, len(categorical_idx_to_uci_codes()))

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.conv3(x)
        # x = F.relu(x)

        x = torch.flatten(x, 1)
        for linear, drop in zip(self.linears, self.dropouts):
            x = linear(x)
            x = F.relu(x)
            x = drop(x)

        output = F.log_softmax(x, dim=1)
        return output


def predict_next_move(model: nn.Module, device: torch.device, board: Board, int_to_move: dict[int, list[str]]):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    torch_board = torch.from_numpy(board_matrix).float().to(device)

    predictions = model(torch_board)[0]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = torch.argsort(predictions, descending=True)
    for move_index in sorted_indices:
        possible_moves = int_to_move[int(move_index)]
        for move in possible_moves:
            if move in legal_moves_uci:
                return move
    return None


def chess_model_player_func(model: nn.Module, device: torch.device) -> PLAYER_FUNC:
    int_to_move = categorical_idx_to_uci_codes()

    def play_move(board: Board) -> str:
        my_move = predict_next_move(model, device, board, int_to_move)
        return my_move

    return play_move


N_DATASET_TRAINS = 1000

if __name__ == '__main__':
    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
    checkpoint_dir = Path("/home/garrickw/rl_learning/weela_chess/data/checkpoints/torch_streaming")
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_data_kwargs = {
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        'shuffle': False
    }
    dataset_factory = TorchMoveDataLoaderFactory(pgn_np_move_streamer(db_path, first_n_moves_only=30,
                                                                      repeat=False), dataset_size=5_000, **train_data_kwargs)

    torch.manual_seed(seed=42)
    device = torch.device("cuda")
    model = Net().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} params")

    player_func = chess_model_player_func(model, device)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    patience = 10
    counter = 0
    best_model_state, best_loss_so_far = model.state_dict(), 9999999

    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    with dataset_factory as data_fact:
        for i, ds_num in enumerate(tqdm(range(N_DATASET_TRAINS))):
            dataset = data_fact.get_next_move_dataloader()
            for j in range(min(i + 1, 10)):
                pytorch_train(model, device, train_loader=dataset, optimizer=optimizer)
            train_loss = pytorch_test(model, device, test_loader=dataset)
            print(f"Train loss: {train_loss}")

            if i % 5 == 0:
                torch.save(model.state_dict(), checkpoint_dir / f"torch_chess_model_epch_{i}.pt")

            test_ds = data_fact.get_next_move_dataloader()
            test_loss = pytorch_test(model, device, test_loader=test_ds)
            print(f"Test loss: {test_loss}")

            if test_loss < best_loss_so_far:
                best_model_state = model.state_dict()
                best_loss_so_far = test_loss
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    break

    torch.save(model.state_dict(), checkpoint_dir / f"torch_chess_model_epch_{i}.pt")

    # # n_moves, i_won = asyncio.run(play_against_stockfish(player_func))
    # n_moves, i_won = loop.run_until_complete(play_against_stockfish(player_func))
    # if i_won:
    #     print(f"this will not happen")
    # else:
    #     print(f"lasted {n_moves} moves against stockfish")
    # loop.close()
    # throws an error for no good reason
