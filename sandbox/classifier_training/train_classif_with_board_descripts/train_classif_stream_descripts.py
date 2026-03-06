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
from weela_chess.models.board_descript_dnn_classif import ClassifDescNet

N_DATASET_TRAINS = 1000

if __name__ == '__main__':
    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
    board_desc_path = Path("/home/garrickw/rl_learning/weela_chess/data/LichessEliteMixtralBoardDescripts")
    checkpoint_dir = Path("/home/garrickw/rl_learning/weela_chess/data/checkpoints/mixtral_augment")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_data_kwargs = {
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        'shuffle': False
    }
    raw_inputs = pgn_np_move_streamer(db_path)
    raw_input_iter = prepare_move_descript_io_iter(raw_inputs, board_descript_iter(board_desc_path), embed_model="BAAI/bge-small-en-v1.5")
    dataset_generator = TorchDataLoaderPreloader(raw_input_iter=raw_input_iter, dataset_size=100, use_threads=True, **train_data_kwargs)

    torch.manual_seed(seed=42)
    device = torch.device("cuda")
    model = ClassifDescNet().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} params")

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    patience = 10
    counter = 0
    best_model_state, best_loss_so_far = model.state_dict(), 9999999

    with dataset_generator as data_fact:
        for i, ds_num in enumerate(tqdm(range(N_DATASET_TRAINS))):
            dataset = data_fact.get_next_dataloader()
            for j in range(min(i + 1, 10)):
                pytorch_train(model, device, train_loader=dataset, optimizer=optimizer)
            train_loss = pytorch_test(model, device, test_loader=dataset)
            print(f"Train loss: {train_loss}")

            if i % 5 == 0:
                torch.save(model.state_dict(), checkpoint_dir / f"torch_chess_model_epch_{i}.pt")

            test_ds = data_fact.get_next_dataloader()
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

    torch.save(best_model_state, checkpoint_dir / f"torch_chess_model_final.pt")

    # player_func = chess_model_player_func(model, device)
    # # n_moves, i_won = asyncio.run(play_against_stockfish(player_func))
    # n_moves, i_won = loop.run_until_complete(play_against_stockfish(player_func))
    # if i_won:
    #     print(f"this will not happen")
    # else:
    #     print(f"lasted {n_moves} moves against stockfish")
    # loop.close()
    # throws an error for no good reason
