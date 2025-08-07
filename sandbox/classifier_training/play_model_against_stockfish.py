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
    from weela_chess.datasets.torch_move_dataset_factory import TorchMoveDataLoaderFactory
    return (
        Board,
        F,
        PLAYER_FUNC,
        Path,
        board_to_matrix,
        categorical_idx_to_uci_codes,
        nn,
        play_against_stockfish_iter,
        torch,
    )


@app.cell
def _(F, categorical_idx_to_uci_codes, nn, torch):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # self.conv1 = nn.Conv2d(8, 64, 3, 1)
            # self.conv2 = nn.Conv2d(64, 128, 3, 1)
            # self.conv3 = nn.Conv2d(128, 128, 3, 1)
            # self.dropout1 = nn.Dropout(0.25)
            # self.dropout2 = nn.Dropout(0.5)

            self.dropouts = nn.ModuleList([
                nn.Dropout(0.0),
                nn.Dropout(0.0),
                nn.Dropout(0.0),
                nn.Dropout(0.15)
            ])

            self.linears = nn.ModuleList([
                nn.Linear(768, 768),
                nn.Linear(768, 768),
                nn.Linear(768, 256),
                nn.Linear(256, len(categorical_idx_to_uci_codes()))
            ])

            # self.fc1 = nn.Linear(1536, 1536)
            # self.fc2 = nn.Linear(1536, 256)
            # self.fc3 = nn.Linear(256, len(categorical_idx_to_uci_codes()))

        def forward(self, x):
            # x = self.conv1(x)
            # x = F.relu(x)
            # x = self.conv2(x)
            # x = F.relu(x)
            # x = self.conv3(x)
            # x = F.relu(x)

            x = torch.flatten(x, 1)
            for linear, drop in zip(self.linears, self.dropouts):
                x = linear(x)
                x = F.relu(x)
                x = drop(x)

            # x = self.fc1(x)
            # x = F.relu(x)
            # x = self.fc2(x)
            # x = F.relu(x)
            # x = self.fc3(x)
            output = F.log_softmax(x, dim=1)
            return output
    return (Net,)


@app.cell
def _(
    Board,
    PLAYER_FUNC,
    board_to_matrix,
    categorical_idx_to_uci_codes,
    nn,
    torch,
):
    def predict_next_move(model: nn.Module, device: torch.device, board: Board, int_to_move: dict[int, list[str]]):
        board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
        torch_board = torch.from_numpy(board_matrix).float().to(device)

        predictions = model(torch_board)[0]
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        sorted_indices = torch.argsort(predictions, descending=True)
        for move_index in sorted_indices:
            possible_moves = int_to_move[int(move_index)]
            for move in possible_moves:
                if move in legal_moves_uci:
                    return move
        return None

    def chess_model_player_func(model: nn.Module, device: torch.device) -> PLAYER_FUNC:
        int_to_move = categorical_idx_to_uci_codes()

        def play_move(board: Board) -> str:
            my_move = predict_next_move(model, device, board, int_to_move)
            return my_move

        return play_move
    return (chess_model_player_func,)


@app.cell
def _(Net, Path, chess_model_player_func, torch):
    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
    checkpoint_dir = Path("/home/garrickw/rl_learning/weela_chess/data/checkpoints/torch_streaming")

    # torch.load(checkpoint_dir / f"torch_chess_model_epch_5.pt", weights_only=False)
    device = torch.device("cuda")
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_dir / f"torch_chess_model_epch_15.pt", weights_only=True))
    model.eval()

    player_func = chess_model_player_func(model, device)
    return (player_func,)


@app.cell
def _():
    #board = Board()
    #board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    #torch_board = torch.from_numpy(board_matrix).float().to(device)
    #
    #predictions = model(torch_board)[0]
    #torch.argsort(predictions, descending=True)
    # _player_func = chess_model_player_func(model, device)

    return


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
