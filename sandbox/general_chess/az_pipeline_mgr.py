import os
import pickle
from pathlib import Path
import uuid
from pydantic import BaseModel
import torch
import json
import numpy as np
import chess

from sandbox.general_chess.pipeline_models import PlayDatasetDescriptor, ChessPlayDatasetDescriptor
from weela_chess.alphazero.alpha_zero_fn_approach import two_player_mcts_play, TurnState
from weela_chess.alphazero.chess.chess_mcts_state_machine import ChessWinOrLossEvaluator, MinimalChessStateEncoder, ChessMctsStateMachine
from weela_chess.alphazero.chess.chess_res_net import ChessResNetDNA, ChessResNet
from weela_chess.alphazero.chess.stockfish_utils import stockfish_player_fn, stockfish_eval_fn
from weela_chess.alphazero.mcts import MCTS, MCTSConfig, build_torch_mcts_model, MCTSModel
from weela_chess.project_files import RESULTS_DIR


class ModelDescriptor(BaseModel):
    model_id: str
    architecture_id: str | None = None

    train_dataset_history: list[str] = []


class ChessResNetDescriptor(ModelDescriptor):
    architecture_id: str = "res_net"
    genome: ChessResNetDNA


class AlphaZeroFileMgr:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

        self.models_dir = self.project_dir / "models"
        os.makedirs(self.models_dir, exist_ok=True)

        self.transient_dataset_dir = self.project_dir / "transient_datasets"
        os.makedirs(self.transient_dataset_dir, exist_ok=True)

        self.dataset_dir = self.project_dir / "datasets"
        os.makedirs(self.dataset_dir, exist_ok=True)

    def list_play_datasets(self) -> list[str]:
        pass


class ChessAlphaZeroPipelineMgr:

    def __init__(self, results_dir: Path, stockfish_engine: chess.engine.SimpleEngine):
        self.results_dir = results_dir
        # chess.engine.SimpleEngine.popen_uci("/home/garrickw/rl_learning/weela_chess/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")
        self.stockfish_engine = stockfish_engine

        self.project_file_mgr: AlphaZeroFileMgr | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_project(self, project_name: str):
        self.project_file_mgr = AlphaZeroFileMgr(self.results_dir / project_name)

    def load_all_model_descripts(self) -> list[ModelDescriptor]:
        descripts = []
        for model_dir in self.project_file_mgr.models_dir.iterdir():
            descript_file = (model_dir / "model_descript.json")
            if descript_file.exists():
                descripts.append(ModelDescriptor(**json.loads(descript_file.read_text())))
        return descripts

    def load_model_descript(self, model_id: str) -> ModelDescriptor | ChessResNetDescriptor:
        model_descript_file = self.project_file_mgr.models_dir / model_id / "model_descript.json"
        model_descript = ModelDescriptor(**json.loads(model_descript_file.read_text()))
        if model_descript.architecture_id == "res_net":
            model_descript = ChessResNetDescriptor(**json.loads(model_descript_file.read_text()))
        return model_descript

    def prepare_stockfish_model(self) -> ModelDescriptor:
        model_descripts = self.load_all_model_descripts()
        stockfish_models = [x for x in model_descripts if x.architecture_id == "stockfish"]
        if len(stockfish_models) > 0:
            return stockfish_models[0]
        model_descript = ModelDescriptor(
            model_id=str(uuid.uuid4()),
            architecture_id="stockfish"
        )
        model_dir = self.project_file_mgr.models_dir / model_descript.model_id
        os.makedirs(model_dir, exist_ok=True)

        descript_file = (model_dir / "model_descript.json")
        descript_file.write_text(json.dumps(model_descript.model_dump(mode="json")))

        return model_descript

    def prepare_res_net_model(self, genome: ChessResNetDNA) -> ModelDescriptor:
        model = ChessResNet(genome, self.device)
        model_id = str(uuid.uuid4())
        model_dir = self.project_file_mgr.models_dir / model_id
        os.makedirs(model_dir, exist_ok=True)

        torch.save(model.state_dict(), model_dir / f"model_state.pt")

        model_descript = ChessResNetDescriptor(
            model_id=model_id,
            genome=genome
        )
        (model_dir / "model_descript.json").write_text(json.dumps(model_descript.model_dump(mode="json")))
        return model_descript

    def load_model(self, model_id: str) -> torch.nn.Module:
        model_file = self.project_file_mgr.models_dir / model_id / "model_state.pt"
        model_descript = self.load_model_descript(model_id)
        # must be a resnet for now (only nn type type)
        model = ChessResNet(model_descript.genome, self.device)
        model.load_state_dict(torch.load(model_file))
        return model

    def load_mcts_model(self, model_id: str) -> MCTSModel:
        model_descript = self.load_model_descript(model_id)
        if model_descript.architecture_id == "stockfish":
            return stockfish_eval_fn(self.stockfish_engine, limit_sec=0.1,
                                     uniform_policy=True)
        else:
            model = self.load_model(model_id)
            return build_torch_mcts_model(model, MinimalChessStateEncoder(), self.device)
