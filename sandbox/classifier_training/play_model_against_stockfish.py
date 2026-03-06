import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import asyncio
    import time
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from chess import Board
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import TensorDataset
    from tqdm import tqdm
    import numpy as np

    from torch_utils.torch_train_utils import pytorch_train, pytorch_test
    from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves, categorical_idx_to_uci_codes
    from weela_chess.chess_utils.stockfish_eval import PLAYER_FUNC, play_against_stockfish, play_against_stockfish_iter
    from weela_chess.data_io.pgn_db_utils import pgn_np_move_streamer, board_to_matrix
    return (
        Board,
        PLAYER_FUNC,
        Path,
        categorical_idx_to_uci_codes,
        nn,
        play_against_stockfish_iter,
        torch,
    )


@app.cell
def _(
    Board,
    PLAYER_FUNC,
    categorical_idx_to_uci_codes,
    nn,
    predict_next_move,
    torch,
):
    def chess_model_player_func(model: nn.Module, device: torch.device) -> PLAYER_FUNC:
        int_to_move = categorical_idx_to_uci_codes()

        def play_move(board: Board) -> str:
            my_move = predict_next_move(model, device, board, int_to_move)
            return my_move

        return play_move
    return (chess_model_player_func,)


@app.cell
def _(Path, chess_model_player_func, torch):
    # from weela_chess.models.board_only_dnn_classif import Net
    from weela_chess.models.board_descript_dnn_classif import ClassifDescNet as Net

    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
    board_desc_path = Path("/home/garrickw/rl_learning/weela_chess/data/LichessEliteMixtralBoardDescripts")
    # checkpoint_dir = Path("/home/garrickw/rl_learning/weela_chess/data/checkpoints/board_only_classif/")
    checkpoint_dir = Path("/home/garrickw/rl_learning/weela_chess/data/checkpoints/mixtral_augment/")

    device = torch.device("cuda")
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_dir / f"torch_chess_model_final.pt", weights_only=True))
    model.eval()

    player_func = chess_model_player_func(model, device)
    return (player_func,)


@app.cell
def _(play_against_stockfish_iter, player_func):
    game_iter = play_against_stockfish_iter(player_func)
    return (game_iter,)


@app.cell
async def _(game_iter):
    game_state = await game_iter.__anext__()
    game_state
    return


if __name__ == "__main__":
    app.run()
