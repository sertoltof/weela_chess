import asyncio
import sys

# sys.path.append("../../")

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# from keras.api.layers import Flatten, Conv2D, Dense
from keras.api.models import Sequential, load_model
# from keras.api.optimizers import Adam

from chess import Board
from chess import pgn
from chess.pgn import Game

# import tensorflow as tf

from weela_chess.data_io.convert_pgn_to_np import board_to_matrix
import chess.engine


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


async def main():
    model = load_model(
        "/home/gerk/sts_after_images/weela_chess_recreate/src/weela_chess/nick_youtube_tute/chess_model.keras")

    transport, engine = await chess.engine.popen_uci(
        "/home/gerk/sts_after_images/weela_chess_recreate/sandbox/stockfish/stockfish-ubuntu-x86-64-avx2")
    board = Board()
    limit = chess.engine.Limit(time=0.2)

    move_num = 0
    while not board.is_game_over():
        print(f"Making my move num {move_num}")
        next_move = predict_next_move(model, board, int_to_move)
        board.push_uci(next_move)

        #pshhh, as if
        if board.is_game_over():
            break

        print(f"Stockfish moving num {move_num}")
        next_stockfish_move = await engine.play(board, limit)
        board.push(next_stockfish_move.move)
        move_num += 1
    await engine.quit()


if __name__ == '__main__':
    data_dir = Path("/home/gerk/sts_after_images/lichess_elite_np_db")

    int_to_move = json.loads((data_dir / "int_to_move.json").read_text())
    int_to_move = {int(k): v for k, v in int_to_move.items()}

    asyncio.run(main())
