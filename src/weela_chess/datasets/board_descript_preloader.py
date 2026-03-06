from more_itertools import chunked

from weela_chess.datasets.torch_dataloader_preloader import TorchDataLoaderPreloader
import numpy as np
import json
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Iterable, TypeVar, Generic, Callable
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

from weela_chess.data_io.pgn_db_utils import MOVE_STATE, pgn_np_move_streamer

# MOVE_WITH_DESCS = tuple[tuple[np.ndarray, str], np.ndarray]
# """Board state, board state description, ohe move uci"""

# def move_descript_dataset_prep_worker(input_iter: Iterable[MOVE_WITH_DESCS],
#                                       dataset_size: int,
#                                       output_q: 'Queue[tuple[Tensor, ...]]',
#                                       embed_model: BaseEmbedding = None):
#     if embed_model is None:
#         embed_model = "BAAI/bge-base-en-v1.5"
#     embedder = SentenceTransformer(embed_model)
#
#     upcoming_x, upcoming_descripts, upcoming_y = [], [], []
#     for (board_state, board_desc), move_code in input_iter:
#         if len(upcoming_x) >= dataset_size:
#             upcoming_x_tens, upcoming_desc_tens = torch.from_numpy(np.array(upcoming_x)).float(), torch.from_numpy(embedder.encode(upcoming_descripts)).float()
#             upcoming_y_tens = torch.from_numpy(np.array(upcoming_y)).long()
#             output_q.put((upcoming_x_tens, upcoming_desc_tens, upcoming_y_tens), block=True)
#             upcoming_x.clear()
#             upcoming_descripts.clear()
#             upcoming_y.clear()
#         upcoming_x.append(board_state)
#         upcoming_descripts.append(board_desc)
#         upcoming_y.append(move_code)

def board_descript_iter(board_desc_path: Path) -> Iterable[str]:
    file_name_to_number = {}
    for game_file in board_desc_path.iterdir():
        if game_file.suffix != ".json":
            continue
        file_num = game_file.stem.split("_")[1]
        file_name_to_number[game_file] = file_num

    sorted_files = sorted(file_name_to_number.keys(), key=lambda k: file_name_to_number[k])
    for game_file in sorted_files:
        game_descripts = json.loads(game_file.read_text())
        for game_descript in game_descripts:
            yield game_descript["SALIENT"]


def prepare_move_descript_io_iter(board_move_iter: Iterable[tuple[np.ndarray, np.ndarray]], descript_iter: Iterable[str],
                                  embed_model: str = None) -> Iterable[tuple[Tensor, Tensor, Tensor]]:
    if embed_model is None:
        embed_model = "BAAI/bge-base-en-v1.5"
    embedder = SentenceTransformer(embed_model)

    # rearranges the tuple such that it is in the format (input, output) where input is the board_state, description tuple
    for data_chunk in chunked(zip(board_move_iter, descript_iter), 200):
        board_state_chunk, move_chunk, descr_chunk = [], [], []
        for (board_state, move_uci), descript in data_chunk:
            board_state_chunk.append(board_state)
            move_chunk.append(move_uci)
            descr_chunk.append(descript)

        board_state_tens = torch.from_numpy(np.array(board_state_chunk)).float()
        move_tens = torch.from_numpy(np.array(move_chunk)).long()
        descr_tens = torch.from_numpy(embedder.encode(descr_chunk)).float()

        for bs, move, descr in zip(board_state_tens, move_tens, descr_tens):
            yield bs, descr, move
