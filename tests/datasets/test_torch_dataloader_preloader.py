from functools import partial
from pathlib import Path

from weela_chess.data_io.pgn_db_utils import pgn_np_move_streamer
from weela_chess.datasets.board_descript_preloader import prepare_move_descript_io_iter, board_descript_iter
from weela_chess.datasets.move_only_preloader import prepare_move_only_tensors
from weela_chess.datasets.torch_dataloader_preloader import TorchDataLoaderPreloader


def test_move_only_dataset():
    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")

    raw_inputs = pgn_np_move_streamer(db_path)

    raw_input_iter = prepare_move_only_tensors(raw_inputs)

    train_data_kwargs = {
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        'shuffle': False
    }
    dataset_generator = TorchDataLoaderPreloader(raw_input_iter=raw_input_iter, dataset_size=100, **train_data_kwargs)
    with dataset_generator as data_fact:
        next = data_fact.get_next_dataloader()
        assert len(next.dataset.tensors) == 2


def test_board_and_descr_dataset():
    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
    board_desc_path = Path("/home/garrickw/rl_learning/weela_chess/data/LichessEliteMixtralBoardDescripts")

    raw_inputs = pgn_np_move_streamer(db_path)
    raw_input_iter = prepare_move_descript_io_iter(raw_inputs, board_descript_iter(board_desc_path), embed_model="BAAI/bge-small-en-v1.5")
    # input_processor = partial(move_descript_dataset_prep_worker, embed_model="BAAI/bge-small-en-v1.5")

    train_data_kwargs = {
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        'shuffle': False
    }
    dataset_generator = TorchDataLoaderPreloader(raw_input_iter=raw_input_iter, dataset_size=100, **train_data_kwargs)
    with dataset_generator as data_fact:
        next = data_fact.get_next_dataloader()
        assert len(next.dataset.tensors) == 3


def test_threaded_board_and_descr_dataset():
    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
    board_desc_path = Path("/home/garrickw/rl_learning/weela_chess/data/LichessEliteMixtralBoardDescripts")

    raw_inputs = pgn_np_move_streamer(db_path)
    raw_input_iter = prepare_move_descript_io_iter(raw_inputs, board_descript_iter(board_desc_path), embed_model="BAAI/bge-small-en-v1.5")
    # input_processor = partial(move_descript_dataset_prep_worker, embed_model="BAAI/bge-small-en-v1.5")

    train_data_kwargs = {
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        'shuffle': False
    }
    dataset_generator = TorchDataLoaderPreloader(raw_input_iter=raw_input_iter, dataset_size=100, use_threads=True,
                                                 **train_data_kwargs)
    with dataset_generator as data_fact:
        next = data_fact.get_next_dataloader()
        assert len(next.dataset.tensors) == 3