import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import asyncio
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
    from weela_chess.chess_utils.stockfish_eval import PLAYER_FUNC, play_against_stockfish, play_against_stockfish_iter
    from weela_chess.data_io.pgn_db_utils import pgn_np_move_streamer, board_to_matrix
    return (Path,)


@app.cell
def _(Path):
    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
    board_descript_path = Path("/home/garrickw/rl_learning/weela_chess/data/LichessEliteMixtralBoardDescripts")
    # checkpoint_dir = Path("/home/garrickw/rl_learning/weela_chess/data/checkpoints/board_only_classif/")
    checkpoint_dir = Path("/home/garrickw/rl_learning/weela_chess/data/checkpoints/mixtral_augment/")
    return board_descript_path, db_path


@app.cell
def _():
    from weela_chess.data_io.ollama_board_state_converter import OllamaBoardStateConverter
    board_converter = OllamaBoardStateConverter(ollama_model_name="gemma2:2b")
    return


@app.cell
def _(board_descript_path, db_path):
    from weela_chess.data_io.pgn_db_utils import pgn_file_streamer, pgn_move_streamer
    from weela_chess.datasets.board_descript_preloader import prepare_move_descript_io_iter, board_descript_iter
    from weela_chess.data_io.ollama_board_state_converter import piece_map_key

    board_state_iter = pgn_move_streamer(db_path)
    descript_iter = board_descript_iter(board_descript_path)
    combined_iter = zip(board_state_iter, descript_iter)
    return (combined_iter,)


@app.cell
def _(combined_iter):
    board, descript = next(combined_iter)
    print(descript)
    board
    return


@app.cell
def _():
    print()
    return


if __name__ == "__main__":
    app.run()
