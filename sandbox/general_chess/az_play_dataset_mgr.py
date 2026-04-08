import json
import os
import pickle
import shutil
import uuid
from typing import Literal

import loguru
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from sandbox.general_chess.pipeline_models import ChessPlayDatasetDescriptor
from weela_chess.alphazero.alpha_zero_fn_approach import two_player_mcts_play, TurnState, self_mcts_play
from weela_chess.alphazero.aux_mcts_interfaces import StateEvaluator, StateEncoder
from weela_chess.alphazero.chess.chess_mcts_state_machine import ChessMctsStateMachine
from weela_chess.alphazero.chess.stockfish_utils import stockfish_player_fn, StockfishEvaluator
from weela_chess.alphazero.mcts import MCTS, MCTSConfig, build_torch_mcts_model

from sandbox.general_chess.az_pipeline_mgr import ChessAlphaZeroPipelineMgr, AlphaZeroFileMgr
from sandbox.general_chess.pipeline_models import PlayDatasetDescriptor
from weela_chess.alphazero.chess.chess_mcts_state_machine import ChessWinOrLossEvaluator, MinimalChessStateEncoder
from weela_chess.models.np_basemodel_support import NumpyArray


class ChessPlayDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    game_results: list[list[TurnState]]
    game_encodings: NumpyArray
    descriptor: ChessPlayDatasetDescriptor


class ChessAlphaZeroPlayDatasetMgr:

    def __init__(self, pipeline_mgr: ChessAlphaZeroPipelineMgr):
        self.pipeline_mgr = pipeline_mgr

        self.player_two = stockfish_player_fn(self.pipeline_mgr.stockfish_engine, limit_sec=0.3)

    @property
    def device(self) -> torch.device:
        return self.pipeline_mgr.device

    @property
    def project_file_mgr(self) -> AlphaZeroFileMgr:
        return self.pipeline_mgr.project_file_mgr

    @property
    def win_loss_evaluator(self) -> StateEvaluator:
        return ChessWinOrLossEvaluator()

    @property
    def stockfish_evaluator(self) -> StateEvaluator:
        return StockfishEvaluator(self.pipeline_mgr.stockfish_engine, limit_sec=0.5, uniform_policy=True)

    @property
    def minimal_encoder(self) -> StateEncoder:
        return MinimalChessStateEncoder()

    def prepare_chess_mcts(self, model_id: str) -> MCTS:
        mcts_config = MCTSConfig(
            c=2.5,

            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3
        )
        root_state = ChessMctsStateMachine()
        # model = self.pipeline_mgr.load_model(model_id)
        # mcts_model = build_torch_mcts_model(model, self.minimal_encoder, self.device)
        mcts_model = self.pipeline_mgr.load_mcts_model(model_id)

        return MCTS(mcts_config, len(root_state.action_to_uci), mcts_model,
                    terminal_state_evaluator=self.win_loss_evaluator)

    def load_dataset(self, dataset_id: str) -> ChessPlayDataset:
        dataset_dir = self.project_file_mgr.dataset_dir / dataset_id
        with open(dataset_dir / "all_games.pkl", "rb") as f:
            all_games: list[list[TurnState]] = pickle.load(f)
        all_game_encodings = np.load(dataset_dir / f"all_game_encodings.npy")
        descriptor = ChessPlayDatasetDescriptor(**json.loads((dataset_dir / "descriptor.json").read_text()))
        return ChessPlayDataset(
            game_results=all_games,
            game_encodings=all_game_encodings,
            descriptor=descriptor
        )

    def save_transient_dataset(self, games_results: list[list[TurnState]],
                               model_id: str, num_mcts_searches: int, mcts_temp: float,
                               opponent: Literal["self", "stockfish"]) -> ChessPlayDatasetDescriptor:
        dataset_id = str(uuid.uuid4())
        dataset_dir = self.project_file_mgr.transient_dataset_dir / dataset_id
        os.makedirs(dataset_dir, exist_ok=True)

        with open(dataset_dir / f"all_games.pkl", "wb") as f:
            pickle.dump(games_results, f)

        all_game_encodings = []
        for game_num, game_result in enumerate(games_results):
            for turn in game_result:
                all_game_encodings.append(self.minimal_encoder.encode_state(turn.state))
        all_game_encodings = np.array(all_game_encodings, dtype=np.float16)
        all_game_file = dataset_dir / f"all_game_encodings.npy"
        np.save(all_game_file, all_game_encodings)

        descriptor = ChessPlayDatasetDescriptor(
            dataset_id=dataset_id,
            num_plays=len(games_results),
            model_id=model_id,

            evaluator_class="ChessWinOrLossEvaluator",
            encoder_class="MinimalChessStateEncoder",

            num_mcts_searches=num_mcts_searches,
            mcts_temp=mcts_temp,
            opponent=opponent,

            is_transient=True
        )
        (dataset_dir / "descriptor.json").write_text(json.dumps(descriptor.model_dump(mode="json"), indent=4))
        return descriptor

    def generate_mcts_dataset(self, model_id: str,
                              num_mcts_searches: int, mcts_temp: float,
                              num_plays: int) -> PlayDatasetDescriptor:
        mcts = self.prepare_chess_mcts(model_id)

        games_results: list[list[TurnState]] = []
        for i in range(num_plays):
            loguru.logger.debug(f"Starting Self Play Game {i}/{num_plays}")
            game_result = self_mcts_play(ChessMctsStateMachine(), num_mcts_searches,
                                         mcts, mcts_temp,
                                         self.win_loss_evaluator,
                                         max_moves=8, early_terminate_evaluator=self.stockfish_evaluator)
            games_results.append(game_result)

        return self.save_transient_dataset(games_results,
                                           model_id, num_mcts_searches, mcts_temp,
                                           opponent="self")

    def generate_two_player_mcts_dataset(self, model_id: str, num_mcts_searches: int, mcts_temp: float,
                                         num_plays: int) -> PlayDatasetDescriptor:
        mcts = self.prepare_chess_mcts(model_id)

        games_results: list[list[TurnState]] = []
        for i in range(num_plays):
            root_state = ChessMctsStateMachine()
            mcts_player = int(np.random.choice([-1, 1]))
            game_result = two_player_mcts_play(root_state, num_searches=num_mcts_searches,
                                               mcts=mcts, mcts_player=mcts_player, mcts_temperature=mcts_temp,
                                               final_state_evaluator=self.win_loss_evaluator,
                                               player_two=self.player_two)
            games_results.append(game_result)

        return self.save_transient_dataset(games_results,
                                           model_id, num_mcts_searches, mcts_temp,
                                           opponent="stockfish")

    def finalize_dataset(self, transient_dataset_id: str):
        from_dir = self.project_file_mgr.transient_dataset_dir / transient_dataset_id
        to_dir = self.project_file_mgr.dataset_dir / transient_dataset_id
        shutil.move(from_dir, to_dir)
