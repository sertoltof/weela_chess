import marimo as mo

import asyncio
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from chess import Board
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np

from torch_utils.torch_train_utils import pytorch_train, pytorch_test
from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves, categorical_idx_to_uci_codes
from weela_chess.chess_utils.stockfish_eval import PLAYER_FUNC, play_against_stockfish, play_against_stockfish_iter
from weela_chess.data_io.ollama_board_state_converter import OllamaBoardStateConverter, load_description_cache_from_pgn_streamers
from weela_chess.data_io.pgn_db_utils import pgn_np_move_streamer, board_to_matrix


def predict_next_move(model: nn.Module, device: torch.device, board: Board, descript_vector: Tensor, int_to_move: dict[int, list[str]]):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    torch_board = torch.from_numpy(board_matrix).float().to(device)
    descript_vector.to(device)

    predictions = model(torch_board, descript_vector)[0]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = torch.argsort(predictions)
    for move_index in sorted_indices:
        possible_moves = int_to_move[int(move_index)]
        for move in possible_moves:
            if move in legal_moves_uci:
                return move
    return None


def torch_descript_model_player_func(model: nn.Module, device: torch.device, board_converter: OllamaBoardStateConverter,
                                     embed_model: str = None) -> PLAYER_FUNC:
    int_to_move = categorical_idx_to_uci_codes()
    if embed_model is None:
        embed_model = "BAAI/bge-base-en-v1.5"
    embedder = SentenceTransformer(embed_model)

    def play_move(board: Board) -> str:
        description = board_converter.describe_board(board)["SALIENT"]
        descript_vector = torch.from_numpy(embedder.encode(description)).float()

        my_move = predict_next_move(model, device, board, descript_vector, int_to_move)
        return my_move

    return play_move

db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
board_desc_path = Path("/home/garrickw/rl_learning/weela_chess/data/LichessEliteMixtralBoardDescripts")
description_cache = load_description_cache_from_pgn_streamers(db_path, board_desc_path)
print()