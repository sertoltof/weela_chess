from itertools import cycle

import numpy as np

from pathlib import Path
from typing import Iterable

from chess.pgn import Game
from chess import pgn, Board

from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_categorical_idx

MOVE_STATE = tuple[np.ndarray, np.ndarray]
"""Board state (8 x 8 x 12) piece positions, uci move code (ohe)"""


def load_pgn(file: Path) -> Iterable[Game]:
    games = []
    with open(file, 'r') as pgn_file:
        while True:
            # print(f"Reading game {len(games)} from file")
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            yield game
    return games


def pgn_file_streamer(pgn_dir: Path, repeat: bool = True,
                      seed: int = 42) -> Iterable[Game]:
    np_rand = np.random.default_rng(seed)

    files = np.array([file for file in pgn_dir.iterdir() if file.name.endswith(".pgn")])
    np_rand.shuffle(files)

    if repeat:
        files = cycle(files)

    for pgn_file in files:
        yield from load_pgn(pgn_file)


def board_to_matrix(board: Board) -> np.ndarray:
    matrix = np.zeros((8, 8, 12), dtype=np.int8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[row, col, piece_type + piece_color] = 1
    return matrix


def game_to_nn_inputs(game: Game, first_n_moves_only: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    all_uci_codes = all_uci_codes_to_categorical_idx()

    x = []
    y = []
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        if first_n_moves_only is not None and i > first_n_moves_only:
            break
        x.append(board_to_matrix(board))
        # move_ohe = np.zeros(len(all_uci_codes))
        # move_ohe[all_uci_codes[move.uci()]] = 1
        y.append(all_uci_codes[move.uci()])
        board.push(move)
    return np.array(x), np.array(y)


def pgn_np_move_streamer(pgn_dir: Path,
                         repeat: bool = True, first_n_moves_only: int | None = None,
                         seed: int = 42) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for game in pgn_file_streamer(pgn_dir, repeat, seed):
        game_states, game_moves = game_to_nn_inputs(game, first_n_moves_only)
        for state, move in zip(game_states, game_moves):
            yield state, move

# db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
# for state, move in pgn_np_move_streamer(db_path):
#     print(state, move)
