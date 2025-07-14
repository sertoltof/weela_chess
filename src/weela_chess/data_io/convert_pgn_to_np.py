import json
import os
import sys
from itertools import chain

import numpy as np
from pathlib import Path

from chess import Board
from chess import pgn
from chess.pgn import Game
# from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
from more_itertools import chunked

from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves

SMALLER_DATASET = False
IS_DEBUGGING = False


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
    matrix = np.zeros((8, 8, 12))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[row, col, piece_type + piece_color] = 1
    return matrix


def games_to_nn_inputs(games: list[Game]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    x = []
    y = []
    for game in tqdm(games):
        board = game.board()
        for move in game.mainline_moves():
            # x.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    move_to_int = {move: idx for idx, move in enumerate(set(y))}
    print(f"There were {len(move_to_int)} different moves detected")
    y = encode_moves(y, move_to_int)
    y = to_categorical(y, num_classes=len(move_to_int))
    return np.array(x), y, move_to_int


def encode_moves(uci_moves: list[str], move_to_int: dict[str, int]) -> np.ndarray:
    return np.array([move_to_int[move] for move in uci_moves])


if __name__ == '__main__':
    data_dir = Path("/home/gerk/sts_after_images/lichess_elite_np_db")
    os.makedirs(data_dir, exist_ok=True)

    dataset_dir = Path(r"/home/gerk/sts_after_images/Lichess_Elite_Database")
    files = [file for file in dataset_dir.iterdir() if file.name.endswith(".pgn")]

    all_games = []
    for pgn_file in tqdm(files):
        games = load_pgn(pgn_file)
        all_games.extend(games)

    x, y, move_to_int = games_to_nn_inputs(all_games)
    all_idxs = list(range(len(x)))
    chunk_size = 7_000 if SMALLER_DATASET else 125_000

    int_to_move = {v:k for k, v in move_to_int.items()}
    (data_dir / "int_to_move.json").write_text(json.dumps(int_to_move))

    all_uci_codes = all_uci_codes_to_moves()
    for move_uci in int_to_move.values():
        if move_uci not in all_uci_codes:
            print("we have an issue")
    print("all moves were valid")

    for i, chunked_idxs in enumerate(chunked(all_idxs, chunk_size)):
        np.save(data_dir / f"elite_db_x_{i}.npy", x[chunked_idxs])
        np.save(data_dir / f"elite_db_min_ohe_y_{i}.npy", y[chunked_idxs])
