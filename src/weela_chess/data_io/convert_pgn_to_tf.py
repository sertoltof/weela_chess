import json
import os
import sys
from functools import partial
from itertools import chain
from typing import Iterable

import numpy as np
from pathlib import Path

from chess import Board
from chess import pgn
from chess.pgn import Game
from tensorflow.python.data import TFRecordDataset
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
from more_itertools import chunked

from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves, all_uci_codes_to_categorical_idx
from weela_chess.data_io.tf_record_utils import save_array_as_tfrecord, parse_tfrecord_meta, parse_tfrecord_full
import tensorflow as tf

SMALLER_DATASET = False
IS_DEBUGGING = True

def stream_pgn_dataset(directory: Path) -> Iterable[Game]:
    files = [file for file in directory.iterdir() if file.name.endswith(".pgn")]
    for pgn_file in tqdm(files):
        with open(pgn_file, 'r') as f:
            while True:
                game = pgn.read_game(f)
                if game is None:
                    break
                yield game


def board_to_matrix(board: Board) -> np.ndarray:
    matrix = np.zeros((8, 8, 12), dtype=np.int8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[row, col, piece_type + piece_color] = 1
    return matrix


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
    return np.array(x), np.array(y)


def encode_moves(uci_moves: list[str], move_to_int: dict[str, int]) -> np.ndarray:
    return np.array([move_to_int[move] for move in uci_moves])


if __name__ == '__main__':
    data_dir = Path("/home/gerk/sts_after_images/lichess_elite_tf_db_debug")
    os.makedirs(data_dir, exist_ok=True)

    dataset_dir = Path(r"/home/gerk/sts_after_images/Lichess_Elite_Database")

    chunk_size = 30 if IS_DEBUGGING else 1_000 if SMALLER_DATASET else 10_000
    games_iter = stream_pgn_dataset(dataset_dir)

    all_uci_codes = all_uci_codes_to_moves()
    # for move_uci in int_to_move.values():
    #     if move_uci not in all_uci_codes:
    #         print("we have an issue")
    for i, game_chunk in enumerate(chunked(games_iter, chunk_size)):
        if i > 60:
            break
        x, y = games_to_nn_inputs(game_chunk)
        meta_x = {"n_elements": len(x)}
        (data_dir / f"elite_db_x_{i}.json").write_text(json.dumps(meta_x))

        save_array_as_tfrecord(x, data_dir / f"elite_db_x_{i}.tfrecord")
        save_array_as_tfrecord(y, data_dir / f"elite_db_min_categorical_y_{i}.tfrecord")

        # raw_dataset = TFRecordDataset(str(data_dir / f"elite_db_x_{i}.tfrecord"))
        # first_feature = list(raw_dataset.take(1))[0]
        # parsed = partial(parse_tfrecord_full, dtype=tf.int8)(first_feature)
        # print()

        # dataset_meta = raw_dataset.map(partial(parse_tfrecord_full, dtype=tf.float64))
        # print(list(dataset_meta.take(1)))
        # print()


    # x, y, move_to_int = games_to_nn_inputs(all_games)
    # all_idxs = list(range(len(x)))
    #
    # int_to_move = {v:k for k, v in move_to_int.items()}
    # (data_dir / "int_to_move.json").write_text(json.dumps(int_to_move))
    #
    #
    # for i, chunked_idxs in enumerate(chunked(all_idxs, chunk_size)):
    #     np.save(data_dir / f"elite_db_x_{i}.npy", x[chunked_idxs])
    #     np.save(data_dir / f"elite_db_min_ohe_y_{i}.npy", y[chunked_idxs])
