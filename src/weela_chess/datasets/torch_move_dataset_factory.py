# factory that outputs one large dataset with multiple batches of moves at a time.
# The idea being that you can train on this one dataset for a while, while the next dataset is being prepped to save on i/o wait times.
# You can also just stream through it at full speed if desired and only look at each move once.
import time
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from weela_chess.data_io.pgn_db_utils import MOVE_STATE, pgn_np_move_streamer


def move_dataset_prep_worker(input_iter: Iterable[MOVE_STATE], dataset_size: int,
                             output_q: 'Queue[tuple[Tensor, Tensor]]'):
    upcoming_x, upcoming_y = [], []
    for board_state, move_code in input_iter:
        if len(upcoming_x) >= dataset_size:
            upcoming_x_tens, upcoming_y_tens = torch.from_numpy(np.array(upcoming_x)).float(), torch.from_numpy(np.array(upcoming_y)).long()
            output_q.put((upcoming_x_tens, upcoming_y_tens), block=True)
            upcoming_x.clear()
            upcoming_y.clear()
        upcoming_x.append(board_state)
        upcoming_y.append(move_code)


class TorchMoveDataLoaderFactory:

    def __init__(self, move_streamer: Iterable[MOVE_STATE], dataset_size: int,
                 **torch_dataloader_kwargs):
        self.move_streamer = move_streamer
        self.dataset_size = dataset_size
        self.torch_dataloader_kwargs = torch_dataloader_kwargs

        self.share_q: Queue[tuple[Tensor, Tensor]] = Queue(maxsize=1)
        # self.worker = MoveDatasetPrepWorker(move_streamer, self.share_q)

    def __enter__(self) -> 'TorchMoveDataLoaderFactory':
        self.worker_proc = Process(target=move_dataset_prep_worker, args=[self.move_streamer, self.dataset_size, self.share_q])
        self.worker_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.worker_proc.kill()
        self.worker_proc.join()
        self.worker_proc.close()

    def get_next_move_dataloader(self) -> DataLoader:
        next_x, next_y = self.share_q.get(block=True)
        # next_x, next_y = torch.from_numpy(next_x).float(), torch.from_numpy(next_x).long()
        next_ds = TensorDataset(next_x, next_y)
        return DataLoader(next_ds, **self.torch_dataloader_kwargs)

# db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
# # for state, move in pgn_np_move_streamer(db_path):
# with TorchMoveDataLoaderFactory(pgn_np_move_streamer(db_path), dataset_size=100) as ds:
#     for i in range(1000):
#         next = ds.get_next_move_dataset()
#         print("got next")
