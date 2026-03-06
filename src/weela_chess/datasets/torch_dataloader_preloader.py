# factory that outputs one large dataset with multiple batches of examples at a time.
# The idea being that you can train on this one dataset for a while, while the next dataset is being prepped to save on i/o wait times.
# You can also just stream through it at full speed if desired and only look at each example once.
import json
import threading
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Queue, Process
from pathlib import Path
from threading import Thread
from typing import Iterable, TypeVar, Generic, Callable
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

from weela_chess.data_io.pgn_db_utils import MOVE_STATE, pgn_np_move_streamer


def lazy_tensor_load(input_iter: Iterable[tuple[Tensor, ...]], dataset_size: int, output_q: 'Queue[tuple[Tensor, ...]]',
                     thread_safe: bool = False):
    tensors_by_idx: dict[int, list[Tensor]] = defaultdict(list)
    for tensors in input_iter:
        if len(tensors_by_idx[0]) >= dataset_size:
            tensor_stacks = []
            for idx, tensor_list in tensors_by_idx.items():
                tensor_stacks.append(torch.stack(tensor_list, dim=0))
            # thread safe because we yield control occasionally, allowing a graceful killing
            if thread_safe:
                while True:
                    try:
                        output_q.put(tuple(tensor_stacks), block=False)
                        break
                    except:
                        pass
                    yield
                    time.sleep(0.05)
            else:
                output_q.put(tuple(tensor_stacks), block=True)
            tensors_by_idx = defaultdict(list)
        for i, tensor in enumerate(tensors):
            tensors_by_idx[i].append(tensor)


def lazy_tensor_load_threaded(input_iter: Iterable[tuple[Tensor, ...]], dataset_size: int, output_q: 'Queue[tuple[Tensor, ...]]',
                              exit_event: threading.Event):
    lazy_tensor_iter = iter(lazy_tensor_load(input_iter, dataset_size, output_q, thread_safe=True))
    while not exit_event.is_set():
        next(lazy_tensor_iter)


class TorchDataLoaderPreloader:

    def __init__(self, raw_input_iter: Iterable[tuple[Tensor, ...]],
                 dataset_size: int, use_threads: bool = False,
                 **torch_dataloader_kwargs):
        self.input_iter = raw_input_iter
        self.dataset_size = dataset_size
        self.use_threads = use_threads

        self.torch_dataloader_kwargs = torch_dataloader_kwargs
        self.share_q: Queue[tuple[Tensor, Tensor]] = Queue(maxsize=1)

    def __enter__(self) -> 'TorchDataLoaderPreloader':
        if self.use_threads:
            self.exit_event = threading.Event()
            self.worker_thread = Thread(target=lazy_tensor_load_threaded, args=[self.input_iter, self.dataset_size, self.share_q, self.exit_event])
            self.worker_thread.start()
        else:
            self.worker_proc = Process(target=lazy_tensor_load, args=[self.input_iter, self.dataset_size, self.share_q])
            self.worker_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_threads:
            self.exit_event.set()
            self.worker_thread.join()
        else:
            self.worker_proc.kill()
            self.worker_proc.join()
            self.worker_proc.close()

    def get_next_dataloader(self) -> DataLoader:
        next_tensors = self.share_q.get(block=True)
        # next_x, next_y = torch.from_numpy(next_x).float(), torch.from_numpy(next_x).long()
        next_ds = TensorDataset(*next_tensors)
        return DataLoader(next_ds, **self.torch_dataloader_kwargs)
