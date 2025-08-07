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

from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves

sys.path.append("../../")

# +
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from functools import partial

from keras.api.layers import Flatten, Conv2D, Dense, InputLayer
from keras.api.models import Sequential, load_model
from keras.api.optimizers import Adam
# from keras.callbacks import EarlyStopping
from keras.src.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.python.data import TFRecordDataset
# tf.config.list_physical_devices()
# tf.debugging.set_log_device_placement(True)
# -

from chess import Board
from chess import pgn
from chess.pgn import Game
import chess.engine

from weela_chess.data_io.convert_pgn_to_np import board_to_matrix
from weela_chess.data_io.tf_record_utils import parse_tfrecord_full

# # Load Files

data_dir = Path("/home/gerk/sts_after_images/lichess_elite_tf_db_debug")

file_names = [str(x) for x in data_dir.iterdir() if "db_x" in x.name and ".tfrecord" in x.name]
meta_jsons = [json.loads(x.read_text()) for x in data_dir.iterdir() if "db_x" in x.name and ".json" in x.name]
y_file_names = [str(x) for x in data_dir.iterdir() if "categorical_y" in x.name]

total_elements = sum([x["n_elements"] for x in meta_jsons])

# file_names = file_names[:1]
# file_names_dataset = tf.data.Dataset.from_tensor_slices(file_names)

input_dataset = TFRecordDataset(file_names)
dataset = input_dataset.map(partial(parse_tfrecord_full, dtype=tf.int8))
dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
first = list(dataset.take(1))[0]
first.shape
# print(list(input_dataset.take(1)))
# first_feature = list(input_dataset.take(1))[0]
# parsed = partial(parse_tfrecord_full, dtype=tf.int8)(first_feature)
# print()

y_dataset = TFRecordDataset(y_file_names)
y_dataset = y_dataset.map(partial(parse_tfrecord_full, dtype=tf.int64))
y_dataset = y_dataset.flat_map(tf.data.Dataset.from_tensor_slices)
first = list(y_dataset.take(1))[0]
first.shape

combined_dataset = tf.data.Dataset.zip((dataset, y_dataset))
combined_dataset = combined_dataset.batch(batch_size=32)
train_slice_idx = int((total_elements // 32) * 0.9)

train_data = combined_dataset.take(train_slice_idx)
validation_data = combined_dataset.skip(train_slice_idx)

# # Define Model

model = Sequential([InputLayer(input_shape=(8, 8, 12)),
    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(128, (3, 3), activation="relu"),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(len(all_uci_codes_to_moves()), activation="softmax")
])


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# # Train Model

early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=10,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the best weights obtained during training
)

dummy_data = tf.data.Dataset.from_tensor_slices(tf.random.uniform((50, 8, 8, 12)))
dummy_labels = tf.data.Dataset.from_tensor_slices(tf.random.uniform((50, len(all_uci_codes_to_moves()))))
dummy_combined = tf.data.Dataset.zip((dummy_data, dummy_labels))

model.fit(train_data,
          epochs=50, callbacks=[early_stopping])
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


# transport, engine = await chess.engine.popen_uci("/home/gerk/sts_after_images/weela_chess_recreate/sandbox/stockfish/stockfish-ubuntu-x86-64-avx2")
# await engine.configure({"Skill Level": 1})
#
# # +
# async def stockfish_game_iter():
#     board = Board()
#     limit = chess.engine.Limit(time=0.1)
#
#     while not board.is_game_over():
#         next_move = predict_next_move(model, board, int_to_move)
#         board.push_uci(next_move)
#         yield board
#
#         #pshhh, as if
#         if board.is_game_over():
#             break
#
#         next_stockfish_move = await engine.play(board, limit)
#         board.push(next_stockfish_move.move)
#         yield board
#
# stockfish_game = stockfish_game_iter()
# # -
#
# next_move = await stockfish_game.__anext__()
# next_move
