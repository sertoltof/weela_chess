import json
import os
import sys
from pathlib import Path

from chess import Piece
from tqdm import tqdm

from weela_chess.data_io.ollama_board_state_converter import OllamaBoardStateConverter
from weela_chess.data_io.pgn_db_utils import pgn_file_streamer

def piece_map_key(piece_map: dict[int, Piece]) -> str:
    key = ""
    for square_num, piece in piece_map.items():
        key += str(square_num)
        key += "w" if piece.color else "b"
        key += str(piece.piece_type)
    return key

def file_name(game_num: int) -> str:
    return f"game_{game_num}.json"

if __name__ == '__main__':
    model_name = sys.argv[1]
    # Path("/home/garrickw/rl_learning/weela_chess/data/Lichess Elite Database")
    db_path = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    os.makedirs(out_dir, exist_ok=True)

    game_iter = pgn_file_streamer(pgn_dir=db_path, seed=None)
    converter = OllamaBoardStateConverter(ollama_model_name=model_name)

    # don't re-describe the exact same piece placements
    description_cache: dict[str, dict[str, str]] = {}

    for game_num, game in enumerate(game_iter):
        game_file = out_dir / file_name(game_num)

        preloaded_descriptions = []
        if game_file.exists():
            preloaded_descriptions = json.loads(game_file.read_text())

        game_state_descripts = []

        board = game.board()
        for move_num, move in enumerate(tqdm(list(game.mainline_moves()))):
            board.push(move)
            state_key = piece_map_key(board.piece_map())
            if game_file.exists():
                board_desc = preloaded_descriptions[move_num]
            else:
                if state_key in description_cache:
                    board_desc = description_cache[state_key]
                else:
                    board_desc = converter.describe_board(board)
                    description_cache[state_key] = board_desc
            game_state_descripts.append(board_desc)
        game_file.write_text(json.dumps(game_state_descripts, indent=4))
