# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import sys
sys.path.append("../../")

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# +
from keras.api.layers import Flatten, Conv2D, Dense
from keras.api.models import Sequential, load_model
from keras.api.optimizers import Adam
from keras.callbacks import EarlyStopping

import tensorflow as tf
# -

from chess import Board
from chess import pgn
from chess.pgn import Game
import chess.engine

from weela_chess.data_io.convert_pgn_to_np import board_to_matrix

tf.config.list_physical_devices()
# tf.debugging.set_log_device_placement(True)

# # Load Files

N_FILES_TO_LOAD = 22
data_dir = Path("/home/gerk/sts_after_images/lichess_elite_np_db")

int_to_move = json.loads((data_dir / "int_to_move.json").read_text())
int_to_move = {int(k):v for k, v in int_to_move.items()}

# +
x, y = [], []
n_moves = None

i = 0
while True:
    if  N_FILES_TO_LOAD is not None and i >= N_FILES_TO_LOAD:
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
# -

# # Define Model

model = Sequential([
    Conv2D(64, (3, 3), activation="relu", input_shape=(8, 8, 12)),
    Conv2D(128, (3, 3), activation="relu"),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(n_moves, activation="softmax")
])


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# # Train Model

early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=10,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the best weights obtained during training
)

model.fit(x, y, epochs=50, validation_split=0.1, batch_size=32, callbacks=[early_stopping])
model.save("./chess_model.keras")

# # Evaluate Model

model = load_model("chess_model.keras")


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


transport, engine = await chess.engine.popen_uci("/home/gerk/sts_after_images/weela_chess_recreate/sandbox/stockfish/stockfish-ubuntu-x86-64-avx2")
await engine.configure({"Skill Level": 1})

# +
async def stockfish_game_iter():
    board = Board()
    limit = chess.engine.Limit(time=0.1)

    while not board.is_game_over():
        next_move = predict_next_move(model, board, int_to_move)
        board.push_uci(next_move)
        yield board

        #pshhh, as if
        if board.is_game_over():
            break
        
        next_stockfish_move = await engine.play(board, limit)
        board.push(next_stockfish_move.move)
        yield board
        
stockfish_game = stockfish_game_iter()
# -

next_move = await stockfish_game.__anext__()
next_move
