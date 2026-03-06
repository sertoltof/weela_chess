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
from weela_chess.data_io.pgn_db_utils import pgn_np_move_streamer, board_to_matrix
from weela_chess.datasets.board_descript_preloader import prepare_move_descript_io_iter, board_descript_iter
from weela_chess.datasets.torch_dataloader_preloader import TorchDataLoaderPreloader

LINEAR_SIZE = 1600


class ClassifDescNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.0),
            nn.Dropout(0.0),
            nn.Dropout(0.0),
            nn.Dropout(0.0),
            nn.Dropout(0.0),
            nn.Dropout(0.15)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(1152, LINEAR_SIZE),
            nn.Linear(LINEAR_SIZE, LINEAR_SIZE),
            nn.Linear(LINEAR_SIZE, LINEAR_SIZE),
            nn.Linear(LINEAR_SIZE, LINEAR_SIZE),
            nn.Linear(LINEAR_SIZE, 256),
            nn.Linear(256, len(categorical_idx_to_uci_codes()))
        ])

    def forward(self, board_state, board_desc):
        board_state, board_desc = torch.flatten(board_state, 1), torch.flatten(board_desc, 1)
        x = torch.concat([board_state, board_desc], dim=1)
        for linear, drop in zip(self.linears, self.dropouts):
            x = linear(x)
            x = F.relu(x)
            x = drop(x)

        output = F.log_softmax(x, dim=1)
        return output
