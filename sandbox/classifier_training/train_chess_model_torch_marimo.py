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
        StepLR,
        TorchMoveDataLoaderFactory,
        board_to_matrix,
        categorical_idx_to_uci_codes,
        nn,
        optim,
        pgn_np_move_streamer,
        play_against_stockfish_iter,
        pytorch_test,
        pytorch_train,
        torch,
        tqdm,
    )


@app.cell
def _(F, categorical_idx_to_uci_codes, nn, torch):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(8, 64, 3, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(4096, 256)
            self.fc2 = nn.Linear(256, len(categorical_idx_to_uci_codes()))

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            # x = F.max_pool2d(x, 2)
            # x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            # x = self.dropout2(x)
            x = self.fc2(x)
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
        sorted_indices = torch.argsort(predictions)
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
def _(
    Net,
    Path,
    StepLR,
    TorchMoveDataLoaderFactory,
    chess_model_player_func,
    optim,
    pgn_np_move_streamer,
    torch,
):
    db_path = Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")

    train_data_kwargs = {
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        'shuffle': False
    }
    dataset_factory = TorchMoveDataLoaderFactory(pgn_np_move_streamer(db_path), dataset_size=5_000, **train_data_kwargs)

    torch.manual_seed(seed=42)
    device = torch.device("cuda")
    model = Net().to(device)
    player_func = chess_model_player_func(model, device)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    return dataset_factory, device, model, optimizer, player_func


@app.cell
def _(
    dataset_factory,
    device,
    model,
    optimizer,
    pytorch_test,
    pytorch_train,
    tqdm,
):
    with dataset_factory as data_fact:
        for ds_num in tqdm(range(50)):
            dataset = data_fact.get_next_move_dataloader()
            pytorch_train(model, device, train_loader=dataset, optimizer=optimizer)

            test_ds = data_fact.get_next_move_dataloader()
            loss = pytorch_test(model, device, test_loader=test_ds)
            print(f"loss: {loss}")
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
