from typing import Literal

from pydantic import BaseModel


class PlayDatasetDescriptor(BaseModel):
    dataset_id: str
    num_plays: int

    model_id: str

    evaluator_class: str
    encoder_class: str

    num_mcts_searches: int
    mcts_temp: float
    opponent: str

    is_transient: bool
    """whether or not the dataset has been accepted yet"""


class ChessPlayDatasetDescriptor(PlayDatasetDescriptor):
    evaluator_class: Literal["ChessWinOrLossEvaluator", "StockfishEval"]
    encoder_class: Literal["MinimalChessStateEncoder"]

    num_mcts_searches: int
    mcts_temp: float
    opponent: Literal["self", "stockfish"]
