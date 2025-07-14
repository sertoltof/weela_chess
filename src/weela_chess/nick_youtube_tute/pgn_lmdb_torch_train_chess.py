# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
import io
import pickle
import sys
from typing import Iterable
import lmdb

sys.path.append("../../")
sys.path.append("../../../")

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, Dataset, DataLoader, IterableDataset

from chess import Board
from chess import pgn
from chess.pgn import Game
import chess.engine

# from weela_chess.data_io.convert_pgn_to_np import board_to_matrix
from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves, all_uci_codes_to_categorical_idx
from weela_chess.data_io.convert_pgn_to_lmdb import game_to_nn_inputs
from sandbox.pytorch_mnist.pytorch_mnist_main import pytorch_train, pytorch_test

# tf.debugging.set_log_device_placement(True)

# # Load Files

print(f"CUDA is Ready: {torch.cuda.is_available()}")

N_FILES_TO_LOAD = 30
data_dir = Path("/home/gerk/sts_after_images/lichess_elite_np_db")

int_to_move = json.loads((data_dir / "int_to_move.json").read_text())
int_to_move = {int(k): v for k, v in int_to_move.items()}


# +
# x, y = [], []
# n_moves = None
# x, y = np.array(x), np.array(y)
# y = np.argmax(y, axis=-1)

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path,
                 train_split_pct: float, is_train: bool = False,
                 transform=None):
        self.lmdb_path = lmdb_path
        # , readahead=False, meminit=False
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get("__keys__".encode()))
            train_split_idx = int(len(self.keys) * (1 - train_split_pct))
            # NOTE: in the future, shuffle these keys before splitting into datasets
            if is_train:
                self.keys = self.keys[:train_split_idx]
            else:
                self.keys = self.keys[train_split_idx:]

            self.length = len(self.keys)
            # self.length = pickle.loads(txn.get(b'__len__'))
            # self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = self.keys[index]
            data = pickle.loads(txn.get(key))

        if self.transform:
            data = self.transform(data)

        return data


class LMDBIterDataset(IterableDataset):
    def __init__(self, random_game_dataset: LMDBDataset,
                 transform=None):
        self.random_dataset = random_game_dataset
        self.transform = transform
        self.observed_length = None

    def __len__(self):
        return self.observed_length

    def random_access_generator(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        all_uci_codes = all_uci_codes_to_categorical_idx()

        game_keys = np.array(list(range(len(self.random_dataset))))
        shuffled_game_keys = np.random.permutation(game_keys)
        observed_n_moves = 0
        for game_key in shuffled_game_keys:
            game_pgn = self.random_dataset[game_key]
            pgn_io = io.StringIO(game_pgn)
            game = chess.pgn.read_game(pgn_io)
            x, y = game_to_nn_inputs(game, all_uci_codes)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            for x_i, y_i in zip(x, y):
                observed_n_moves += 1
                if self.transform:
                    x_i, y_i = self.transform((x_i, y_i))
                yield x_i, y_i
        self.observed_length = observed_n_moves

    def __iter__(self):
        return iter(self.random_access_generator())


# +
def prepare_data(x_y: tuple[torch.Tensor, torch.Tensor]):
    return x_y[0].to(torch.float32), x_y[1][0]

train_dataset_random = LMDBDataset("/home/gerk/sts_after_images/lichess_elite_pgn_lmdb", is_train=True, train_split_pct=0.1)
train_dataset = LMDBIterDataset(train_dataset_random, transform=prepare_data)

test_dataset_random = LMDBDataset("/home/gerk/sts_after_images/lichess_elite_pgn_lmdb", is_train=False, train_split_pct=0.1)
test_dataset = LMDBIterDataset(test_dataset_random, transform=prepare_data)

train_kwargs = {'batch_size': 64}
test_kwargs = {'batch_size': 1000}
cuda_kwargs = {'num_workers': 0,
               'pin_memory': True,
               'shuffle': False}
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)

train_loader = DataLoader(train_dataset, **train_kwargs)
test_loader = DataLoader(test_dataset, **test_kwargs)


# -

# # Define Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, len(all_uci_codes_to_moves()))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


torch.manual_seed(seed=42)
device = torch.device("cuda")
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# # Train Model

patience = 10
counter = 0
best_model_state, best_loss_so_far = model.state_dict(), 9999999

for epoch in range(1, 51):
    torch.save(model.state_dict(), f"torch_chess_model_epoch_{epoch}.pt")
    pytorch_train(model, device, train_loader, optimizer, epoch)
    test_loss = pytorch_test(model, device, test_loader)
    scheduler.step()

    if test_loss < best_loss_so_far:
        best_model_state = model.state_dict()
        best_loss_so_far = test_loss
        counter = 0
    else:
        counter += 1
        if counter > patience:
            break
    torch.save(model.state_dict(), f"torch_chess_model_epoch_{epoch}.pt")

torch.save(best_model_state, "torch_chess_model.pt")

# # Evaluate Model

model = torch.load("torch_chess_model.pt", weights_only=True)


# ## Play a Game

def predict_next_move(model, board, int_to_move):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    predictions = model.predict(board_matrix)[0]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(predictions)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    return None

# transport, engine = await chess.engine.popen_uci("/home/gerk/sts_after_images/weela_chess_recreate/sandbox/stockfish/stockfish-ubuntu-x86-64-avx2")
# await engine.configure({"Skill Level": 1})
#
# # +
# async def stockfish_game_iter():
#     board = Board()
#     limit = chess.engine.Limit(time=0.1)
#
#     while not board.is_game_over():
#         next_move = predict_next_move(model, board, int_to_move)
#         board.push_uci(next_move)
#         yield board
#
#         #pshhh, as if
#         if board.is_game_over():
#             break
#
#         next_stockfish_move = await engine.play(board, limit)
#         board.push(next_stockfish_move.move)
#         yield board
#
# stockfish_game = stockfish_game_iter()
# # -
#
# next_move = await stockfish_game.__anext__()
# next_move
