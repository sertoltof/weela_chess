import io
import json
import os
import pickle
import sys
from itertools import chain
from typing import Iterable

import chess
import numpy as np
from pathlib import Path
import lmdb
import torch
from torchvision import datasets, transforms

from chess import Board
from chess import pgn
from chess.pgn import Game
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from more_itertools import chunked

from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves, all_uci_codes_to_categorical_idx

SMALLER_DATASET = False
IS_DEBUGGING = False


def stream_pgn_dataset(directory: Path) -> Iterable[Game]:
    files = [file for file in directory.iterdir() if file.name.endswith(".pgn")]
    for pgn_file in tqdm(files):
        with open(pgn_file, 'r') as f:
            while True:
                game = pgn.read_game(f)
                if game is None:
                    break
                yield game

def stream_pgn_dataset_strs(directory: Path) -> Iterable[str]:
    files = [file for file in directory.iterdir() if file.name.endswith(".pgn")]
    for pgn_file in tqdm(files):
        with open(pgn_file, 'r') as f:
            while True:
                game = pgn.read_game(f)
                if game is None:
                    break
                exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
                yield game.accept(exporter)

def load_pgn(file: Path) -> list[Game]:
    games = []
    with open(file, 'r') as pgn_file:
        while True:
            print(f"Reading game {len(games)} from file")
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
            if IS_DEBUGGING and len(games) > 5:
                break
            if SMALLER_DATASET and len(games) > 50:
                break
    return games


def board_to_matrix(board: Board) -> np.ndarray:
    matrix = np.zeros((8, 8, 12), dtype=np.int8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[row, col, piece_type + piece_color] = 1
    return matrix

def game_to_nn_inputs(game: Game, uci_codes: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []

    board = game.board()
    for move in game.mainline_moves():
        x.append(board_to_matrix(board))
        y.append(uci_codes[move.uci()])
        board.push(move)
    y = np.array(y)[:, np.newaxis]
    return np.array(x), y

def games_to_nn_inputs(games: list[Game]) -> tuple[np.ndarray, np.ndarray]:
    all_uci_codes = all_uci_codes_to_categorical_idx()

    x = []
    y = []
    for game in tqdm(games):
        board = game.board()
        for move in game.mainline_moves():
            x.append(board_to_matrix(board))
            y.append(all_uci_codes[move.uci()])
            board.push(move)
    # move_to_int = {move: idx for idx, move in enumerate(set(y))}
    # print(f"There were {len(move_to_int)} different moves detected")
    # y = encode_moves(y, move_to_int)
    y = np.array(y)[:, np.newaxis]
    # y = to_categorical(y, num_classes=len(move_to_int))
    return np.array(x), y


def encode_moves(uci_moves: list[str], move_to_int: dict[str, int]) -> np.ndarray:
    return np.array([move_to_int[move] for move in uci_moves])


if __name__ == '__main__':
    data_dir = Path("/home/gerk/sts_after_images/lichess_elite_pgn_lmdb")
    os.makedirs(data_dir, exist_ok=True)

    dataset_dir = Path(r"/home/gerk/sts_after_images/Lichess_Elite_Database")
    # files = [file for file in dataset_dir.iterdir() if file.name.endswith(".pgn")]

    # env = lmdb.open(str(data_dir / 'lichess_elite'), readonly=True, lock=False) # Adjust map_size as needed
    # with env.begin(write=False) as txn:
    #     keys = [f"example_{i}".encode() for i in range(txn.stat()["entries"])]
    #     for key in keys:
    #         data = pickle.loads(txn.get(key))
    #         print()

    # env = lmdb.open(str(data_dir), map_size=(1024 ** 3) * 200)  # Adjust map_size as needed
    # with env.begin(write=True) as txn:
    #     keys = [f"game_{i}".encode() for i in range(txn.stat()["entries"])]
    #     txn.put("__keys__".encode(), pickle.dumps(keys))
    #     n_moves = 0
    #     for key in tqdm(keys):
    #         pgn_io = io.StringIO(pickle.loads(txn.get(key)))
    #         game = chess.pgn.read_game(pgn_io)
    #         # n_moves += len([x for x in game.mainline_moves()])
    #         txn.put("__n_moves__".encode(), pickle.dumps(n_moves))

    env.close()

    log_freq = 1000

    all_uci_codes = all_uci_codes_to_categorical_idx()

    example_num = 0
    env = lmdb.open(str(data_dir), map_size=(1024 ** 3) * 200)  # Adjust map_size as needed
    for i, game_chunk in enumerate(chunked(stream_pgn_dataset_strs(dataset_dir), log_freq)):
        print(f"Converting game chunk: {i}")
        with env.begin(write=True) as txn:
            for game in game_chunk:
                key = f"game_{example_num}"
                example_num += 1
                txn.put(key.encode(), pickle.dumps(game))
    with env.begin(write=True) as txn:
        keys = [f"game_{i}".encode() for i in range(txn.stat()["entries"])]
        txn.put("__keys__".encode(), pickle.dumps(keys))
    env.close()

    # for i, chunked_idxs in enumerate(chunked(all_idxs, chunk_size)):
    #     np.save(data_dir / f"elite_db_x_{i}.npy", x[chunked_idxs])
    #     np.save(data_dir / f"elite_db_min_ohe_y_{i}.npy", y[chunked_idxs])
