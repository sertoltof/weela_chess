import os
import pickle
from pathlib import Path
import uuid

import loguru
from pydantic import BaseModel
import torch
import json
import numpy as np
import chess

from sandbox.general_chess.az_pipeline_mgr import ChessAlphaZeroPipelineMgr
from sandbox.general_chess.az_play_dataset_mgr import ChessAlphaZeroPlayDatasetMgr
from sandbox.general_chess.pipeline_models import PlayDatasetDescriptor, ChessPlayDatasetDescriptor
from weela_chess.alphazero.alpha_zero_fn_approach import two_player_mcts_play, TurnState
from weela_chess.alphazero.chess.chess_mcts_state_machine import ChessWinOrLossEvaluator, MinimalChessStateEncoder, ChessMctsStateMachine
from weela_chess.alphazero.chess.chess_res_net import ChessResNetDNA, ChessResNet
from weela_chess.alphazero.chess.stockfish_utils import stockfish_player_fn, stockfish_eval_fn
from weela_chess.alphazero.mcts import MCTS, MCTSConfig, build_torch_mcts_model, MCTSModel
from weela_chess.project_files import RESULTS_DIR
import sys

if __name__ == '__main__':
    results_dir = RESULTS_DIR
    stockfish_engine = chess.engine.SimpleEngine.popen_uci("/home/garrickw/rl_learning/weela_chess/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")

    try:
        pipeline_mgr = ChessAlphaZeroPipelineMgr(results_dir, stockfish_engine)
        dataset_mgr = ChessAlphaZeroPlayDatasetMgr(pipeline_mgr)

        pipeline_mgr.select_project("self_play_stockfish_value_model")

        # genome = ChessResNetDNA(
        #     num_blocks=5,
        #     num_channels=40,
        #
        #     num_input_channels=17
        # )
        # model_descript = pipeline_mgr.prepare_res_net_model(genome)

        model_descript = pipeline_mgr.prepare_stockfish_model()
        loguru.logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")

        dataset_descript = dataset_mgr.generate_mcts_dataset(model_descript.model_id,
                                                             num_mcts_searches=60, mcts_temp=1.25,
                                                             num_plays=1)
        dataset_mgr.finalize_dataset(dataset_descript.dataset_id)

    finally:
        stockfish_engine.quit()
