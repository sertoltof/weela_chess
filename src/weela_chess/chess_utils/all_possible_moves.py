# https://ai.stackexchange.com/questions/6069/how-do-you-encode-a-chess-move-in-a-neural-network/47973#47973
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Literal, get_args

from chess import Board, PAWN, WHITE, BLACK
from pydantic import BaseModel


class MoveDirections(Enum):
    N = (0, 1)
    NE = (1, 1)
    E = (1, 0)
    SE = (1, -1)
    S = (0, -1)
    SW = (-1, -1)
    W = (-1, 0)
    NW = (-1, 1)

    @staticmethod
    def cardinal_directions() -> list['MoveDirections']:
        return [MoveDirections.N, MoveDirections.E, MoveDirections.S, MoveDirections.W]

    def get_perpendicular_cardinal_directions(self) -> list['MoveDirections']:
        if self in [MoveDirections.N, MoveDirections.S]:
            return [MoveDirections.E, MoveDirections.W]
        if self in [MoveDirections.E, MoveDirections.W]:
            return [MoveDirections.N, MoveDirections.S]


FileNames = Literal["a", "b", "c", "d", "e", "f", "g", "h"]
file_names = list(get_args(FileNames))


def file_rank_to_square_code(file: FileNames, rank: int) -> int:
    rank_offset = (rank - 1) * 8
    file_offset = file_names.index(file)
    return rank_offset + file_offset


class MoveInstruction(BaseModel, ABC):
    move_from_file: FileNames
    move_from_rank: int

    @abstractmethod
    def uci_code(self) -> list[str] | None:
        pass

    @property
    def move_from_file_number(self) -> int:
        return file_names.index(self.move_from_file) + 1


class QueenMoveInstruction(MoveInstruction):
    move_direction: MoveDirections
    move_distance: int

    def uci_code(self) -> list[str] | None:
        x, y = self.move_from_file_number, self.move_from_rank
        x_dir, y_dir = self.move_direction.value
        new_x, new_y = x + (x_dir * self.move_distance), y + (y_dir * self.move_distance)
        if new_x < 1 or new_x > 8 or new_y < 1 or new_y > 8:
            return None
        new_file_name = file_names[new_x - 1]

        is_pawn_white, is_pawn_black = self.move_from_rank == 7, self.move_from_rank == 2
        is_valid_white_pawn_move = self.move_direction in [MoveDirections.N, MoveDirections.NE, MoveDirections.NW]
        is_valid_black_pawn_move = self.move_direction in [MoveDirections.S, MoveDirections.SW, MoveDirections.SE]

        is_white_pawn_moving_forward_one = is_pawn_white and is_valid_white_pawn_move and self.move_distance == 1
        is_black_pawn_moving_forward_one = is_pawn_black and is_valid_black_pawn_move and self.move_distance == 1

        valid_ucis = [f"{self.move_from_file}{self.move_from_rank}{new_file_name}{new_y}"]
        if is_white_pawn_moving_forward_one or is_black_pawn_moving_forward_one:
            valid_ucis.append(f"{self.move_from_file}{self.move_from_rank}{new_file_name}{new_y}q")

        return valid_ucis


class KnightMoveInstruction(MoveInstruction):
    large_step_dir: MoveDirections
    small_step_dir: MoveDirections

    def uci_code(self) -> list[str] | None:
        x, y = self.move_from_file_number, self.move_from_rank

        x_dir, y_dir = self.large_step_dir.value[0] * 2, self.large_step_dir.value[1] * 2
        x_dir_full, y_dir_full = x_dir + self.small_step_dir.value[0], y_dir + self.small_step_dir.value[1]
        new_x, new_y = x + x_dir_full, y + y_dir_full
        if new_x < 1 or new_x > 8 or new_y < 1 or new_y > 8:
            return None
        new_file_name = file_names[new_x - 1]
        return [f"{self.move_from_file}{self.move_from_rank}{new_file_name}{new_y}"]


PawnMoveDirections = Literal["cap_west", "move_up", "cap_east"]
PawnPromotionTypes = Literal["rook", "knight", "bishop"]


class PawnUnderpromoteInstruction(MoveInstruction):
    move_type: PawnMoveDirections
    promote_to: PawnPromotionTypes

    def uci_code(self) -> list[str] | None:
        if self.move_from_rank not in [2, 7]:
            return None

        x, y = self.move_from_file_number, self.move_from_rank
        y_dir = 1 if self.move_from_rank == 7 else -1
        if self.move_type == "cap_west":
            x_dir = -1
        elif self.move_type == "move_up":
            x_dir = 0
        else:
            x_dir = 1

        new_x, new_y = x + x_dir, y + y_dir
        if new_x < 1 or new_x > 8 or new_y < 1 or new_y > 8:
            return None
        new_file_name = file_names[new_x - 1]
        promo_code = "r" if self.promote_to == "rook" else "n" if self.promote_to == "knight" else "b"

        return [f"{self.move_from_file}{self.move_from_rank}{new_file_name}{new_y}{promo_code}"]


def list_all_move_instructions() -> list[MoveInstruction]:
    # files = ["a", "b", "c", "d", "e", "f", "h"]
    ranks = [1, 2, 3, 4, 5, 6, 7, 8]

    # x_y_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    magnitudes = [1, 2, 3, 4, 5, 6, 7]

    moves = []
    for i, file in enumerate(get_args(FileNames)):
        for rank in ranks:
            # hit the 56 queen directions
            for direction in MoveDirections:
                for magnitude in magnitudes:
                    queen_move = QueenMoveInstruction(move_from_file=file, move_from_rank=rank,
                                                      move_direction=direction, move_distance=magnitude)
                    if queen_move.uci_code() is None:
                        continue
                    moves.append(queen_move)

            # hit the 8 knight moves
            for large_step in MoveDirections.cardinal_directions():
                # x_dir, y_dir = large_step.value[0] * 2, large_step.value[1] * 2
                for small_step in large_step.get_perpendicular_cardinal_directions():
                    knight_move = KnightMoveInstruction(move_from_file=file, move_from_rank=rank,
                                                        large_step_dir=large_step, small_step_dir=small_step)
                    if knight_move.uci_code() is None:
                        continue
                    moves.append(knight_move)

            # hit the 9 underpremotions
            for move_dir in get_args(PawnMoveDirections):
                for promo_type in get_args(PawnPromotionTypes):
                    pawn_move = PawnUnderpromoteInstruction(move_from_file=file, move_from_rank=rank,
                                                            move_type=move_dir, promote_to=promo_type)
                    if pawn_move.uci_code() is None:
                        continue
                    moves.append(pawn_move)
    return moves


def all_uci_codes_to_moves() -> dict[str, MoveInstruction]:
    uci_codes_to_moves = {}
    for move in list_all_move_instructions():
        for uci_code in move.uci_code():
            uci_codes_to_moves[uci_code] = move
    return uci_codes_to_moves

def all_uci_codes_to_categorical_idx() -> dict[str, int]:
    uci_codes_to_cat_idxs = {}
    for i, move in enumerate(list_all_move_instructions()):
        for uci_code in move.uci_code():
            uci_codes_to_cat_idxs[uci_code] = i
    return uci_codes_to_cat_idxs

def categorical_idx_to_uci_codes() -> dict[int, list[str]]:
    cat_idxs_to_uci_codes = defaultdict(list)
    for i, move in enumerate(list_all_move_instructions()):
        for uci_code in move.uci_code():
            cat_idxs_to_uci_codes[i].append(uci_code)
    return cat_idxs_to_uci_codes
    # return {v: k for k, v in all_uci_codes_to_categorical_idx().items()}

# all_moves = all_uci_codes_to_categorical_idx()
# idx_to_codes = categorical_idx_to_uci_codes()
# print()