import asyncio
from collections import defaultdict

from chess import Board, PIECE_NAMES
from chess.pgn import Game
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from weela_chess.chess_utils.stockfish_eval import play_stockfish_against_self

col_names = ["a", "b", "c", "d", "e", "f", "g", "h"]


def pawn_state_text(pawn_coords: list[str], color_desc: str) -> str:
    return f"{color_desc} has {len(pawn_coords)} pawns at positions: {','.join(sorted(pawn_coords))}"


def knight_state_text(knight_coords: list[str], color_desc: str) -> str:
    return f"{color_desc} has {len(knight_coords)} knights at positions: {','.join(sorted(knight_coords))}"


def bishop_state_text(bishop_coords: list[str], color_desc: str) -> str:
    return f"{color_desc} has {len(bishop_coords)} bishops at positions: {','.join(sorted(bishop_coords))}"


def rook_state_text(rook_coords: list[str], color_desc: str) -> str:
    return f"{color_desc} has {len(rook_coords)} rooks at positions: {','.join(sorted(rook_coords))}"


def queen_state_text(queen_coords: list[str], color_desc: str) -> str:
    return f"{color_desc} has queen at position(s): {','.join(sorted(queen_coords))}"


def king_state_text(king_coord: str, color_desc: str) -> str:
    return f"{color_desc} has their king at position: {king_coord}"


def piece_state_texts(piece_to_coords: dict[str, list[str]], color_desc: str) -> str:
    piece_state_texts = [
        pawn_state_text(piece_to_coords["pawn"], color_desc),
        knight_state_text(piece_to_coords["knight"], color_desc),
        bishop_state_text(piece_to_coords["bishop"], color_desc),
        rook_state_text(piece_to_coords["rook"], color_desc),
        queen_state_text(piece_to_coords["queen"], color_desc),
        king_state_text(piece_to_coords["king"][0], color_desc),
    ]
    return "\n".join(piece_state_texts)


class OllamaBoardStateConverter:

    def __init__(self, ollama_model_name: str, embed_model: BaseEmbedding = None,
                 **ollama_kwargs):
        if embed_model is None:
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.embed_model = embed_model

        self.ollama_model_name = ollama_model_name
        self.ollama_kwargs = ollama_kwargs

    def describe_board(self, board: Board) -> dict[str, str]:
        active_llm = Ollama(model=self.ollama_model_name, request_timeout=360.0, additional_kwargs=self.ollama_kwargs)
        board_prompt = """You are playing chess. I will describe the positions of each piece on the board and the state of the game.
        Output one sentence describing the most salient feature of the game state. Use the format 'SALIENT: <description>'
        Output a newline character, then a sentence describing the part of the game that the WHITE piece player should focus on. Use the format 'WHITE-FOCUS: <description>'
        Output a newline character, then a sentence describing the part of the game that the BLACK piece player should focus on. Use the format 'BLACK-FOCUS': <description>'"""
        board_prompt_msg = ChatMessage(role=MessageRole.SYSTEM, content=board_prompt)

        white_pieces, black_pieces = defaultdict(list), defaultdict(list)
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            if piece.color:
                white_pieces[PIECE_NAMES[piece.piece_type]].append(f"{col_names[col]}{row + 1}")
            else:
                black_pieces[PIECE_NAMES[piece.piece_type]].append(f"{col_names[col]}{row + 1}")
        board_state_text = f"{piece_state_texts(white_pieces, color_desc='white')}\n\n{piece_state_texts(black_pieces, color_desc='black')}"

        if board.turn:
            board_state_text += "\nIt is white's turn"
        else:
            board_state_text += "\nIt is black's turn"
        board_state_msg = ChatMessage(role=MessageRole.USER, content=board_state_text)
        # could also tell the LLM if either side has castled

        response = active_llm.chat([board_prompt_msg, board_state_msg]).message.content

        response_words = response.split()

        keywords = ["SALIENT", "WHITE-FOCUS", "BLACK-FOCUS"]
        keyword_responses = {k: "" for k in keywords}
        current_keyword, keyword_response = None, []
        for word in response_words:
            matched = False
            for keyword in keywords:
                if word in keyword or keyword in word:
                    if current_keyword is not None:
                        keyword_responses[current_keyword] = " ".join(keyword_response)
                    current_keyword = keyword
                    keyword_response.clear()
                    matched = True
                    break
            if not matched:
                if current_keyword is not None:
                    keyword_response.append(word)
        if len(keyword_response) > 0:
            keyword_responses[current_keyword] = " ".join(keyword_response)
        return keyword_responses


async def testing_converter():
    converter = OllamaBoardStateConverter(ollama_model_name="gemma2")
    async for board_state in play_stockfish_against_self():
        responses = converter.describe_board(board_state)
        print(responses)


if __name__ == '__main__':
    asyncio.run(testing_converter())
