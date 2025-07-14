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

import sys

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
from torch.utils.data import TensorDataset

from chess import Board
from chess import pgn
from chess.pgn import Game
import chess.engine

from weela_chess.data_io.convert_pgn_to_np import board_to_matrix
from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves
# from sandbox.pytorch_mnist.pytorch_mnist_main import pytorch_train, pytorch_test

# tf.debugging.set_log_device_placement(True)

# # Load Files

N_FILES_TO_LOAD = 1
data_dir = Path("/home/gerk/sts_after_images/lichess_elite_np_db")

int_to_move = json.loads((data_dir / "int_to_move.json").read_text())
int_to_move = {int(k): v for k, v in int_to_move.items()}

# +
x, y = [], []
n_moves = None

i = 0
while True:
    if N_FILES_TO_LOAD is not None and i >= N_FILES_TO_LOAD:
        break

    x_file, y_file = Path(data_dir / f"elite_db_x_{i}.npy"), Path(data_dir / f"elite_db_min_ohe_y_{i}.npy")
    if not x_file.exists() or not y_file.exists():
        break
    print(f"loading from file number: {i}")

    x.extend(np.load(x_file))
    y.extend(np.load(y_file))
    n_moves = len(y[0])
    i = i + 1
    print(f"There are now {len(x)} move examples")

x, y = np.array(x), np.array(y)
y = np.argmax(y, axis=-1)

# +
x_tensor = torch.from_numpy(x).float()  # Ensure correct data type (e.g., float32)
y_tensor = torch.from_numpy(y).long()  # For labels, use long

train_slice_idx = int((len(x) * 0.9))

train_kwargs = {'batch_size': 64}
test_kwargs = {'batch_size': 1000}
cuda_kwargs = {'num_workers': 0,
               'pin_memory': True,
               'shuffle': True}
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)

x_train, y_train = x_tensor[:train_slice_idx], y_tensor[:train_slice_idx]
x_test, y_test = x_tensor[train_slice_idx:], y_tensor[train_slice_idx:]

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


# -

# # Define Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, len(all_uci_codes_to_moves()))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# +
def pytorch_train(model, device, train_loader, optimizer, epoch,
                  log_interval = 10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def pytorch_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss
# -

torch.manual_seed(seed=42)
device = torch.device("cuda")
model = Net().to(device)

# model = torch.compile(model)
# pytorch_train = torch.compile(pytorch_train)

optimizer = optim.Adadelta(model.parameters(), lr=1.0)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# # Train Model

patience = 10
counter = 0
best_model_state, best_loss_so_far = model.state_dict(), 9999999

for epoch in range(1, 51):
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
