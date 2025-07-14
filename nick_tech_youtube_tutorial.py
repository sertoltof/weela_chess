import os
from itertools import chain

import numpy as np
from pathlib import Path

from chess import Board
from chess import pgn
from chess.pgn import Game
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm

N_FILES_TO_LOAD = 1

if __name__ == '__main__':
    data_dir = Path("/home/gerk/sts_after_images/lichess_elite_np_db")
    os.makedirs(data_dir, exist_ok=True)

    x, y = [], []

    for i in range(N_FILES_TO_LOAD):
        x_file, y_file = Path(data_dir / f"elite_db_x_{i}.npy"), Path(data_dir / f"elite_db_min_ohe_y_{i}.npy")
        if not x_file.exists() or not y_file.exists():
            break

        x.extend(np.load(x_file))
        y.extend(np.load(y_file))
    