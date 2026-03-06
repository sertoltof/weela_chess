from more_itertools import chunked

import json
from pathlib import Path
from typing import Iterable, TypeVar, Generic, Callable
import numpy as np
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer

def prepare_move_only_tensors(board_move_iter: Iterable[tuple[np.ndarray, np.ndarray]]) -> Iterable[tuple[Tensor, Tensor]]:
    # rearranges the tuple such that it is in the format (input, output) where input is the board_state, description tuple
    for data_chunk in chunked(board_move_iter, 200):
        board_state_chunk, move_chunk = [], []
        for board_state, move_uci in data_chunk:
            board_state_chunk.append(board_state)
            move_chunk.append(move_uci)

        board_state_tens = torch.from_numpy(np.array(board_state_chunk)).float()
        move_tens = torch.from_numpy(np.array(move_chunk)).long()
        for bs, move in zip(board_state_tens, move_tens):
            yield bs, move
