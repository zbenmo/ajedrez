from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generator, Protocol, Tuple, Callable
from collections import Counter
import numpy as np


Square = Tuple[int, int]
GetSquareStatus = Callable[[Square], str]


class Piece(ABC):
    def __init__(self, location: Square, piece: str):
        self.location = location
        self.piece = piece

    @abstractmethod
    def theoretical_moves(
            self, get_square_status: GetSquareStatus) -> Generator[Square, None, None]:
        pass

    def __repr__(self):
        return self.piece


class Pawn(Piece):
    def __init__(self, location: Square, piece: str):
        super().__init__(location, piece)
        direction = +1 if piece == 'p' else -1
        starting_row = 1 if piece == 'p' else 6
        row, col = location
        self.moves = [(row + direction, col)]
        if row == starting_row:
            self.moves.append((row + 2 * direction, col))
        self.captures = []
        if col > 0:
            self.captures.append((row + direction, col - 1))
        if col < 7:
            self.captures.append((row + direction, col + 1))

    def theoretical_moves(
            self, get_square_status: GetSquareStatus) -> Generator[Square, None, None]:
        for move in self.moves:
            if get_square_status(move) == ' ':
                yield move
        for capture in self.captures:
            status = get_square_status(move) 
            if status != ' ' and get_square_status(move).islower() != self.piece.islower():
                yield capture


class RayBased(Piece, ABC):
    def __init__(self, location, piece):
        super().__init__(location, piece)

    def theoretical_moves(
            self, get_square_status: GetSquareStatus) -> Generator[Square, None, None]:
        for ray in self.rays:
            for move in ray:
                status = get_square_status(move)
                if status == ' ':
                    yield move
                else:
                    if status.islower() != self.piece.islower():
                        yield move # capture
                    break # this ray reached a piece


class Queen(RayBased):
    def __init__(self, location: Square, piece: str):
        super().__init__(location, piece)
        row, col = location
        self.rays = [
            [(row, c) for c in range(col + 1, 8)],
            [(row, c) for c in range(col - 1, -1, -1)],
            [(r, col) for r in range(row + 1, 8)],
            [(r, col) for r in range(row - 1, -1, -1)],
            [(r, c) for r, c in zip(range(row + 1, 8), range(col + 1, 8))],
            [(r, c) for r, c in zip(range(row - 1, -1, -1), range(col + 1, 8))],
            [(r, c) for r, c in zip(range(row + 1, 8), range(col - 1, -1, -1))],
            [(r, c) for r, c in zip(range(row - 1, -1, -1), range(col - 1, -1, -1))],
        ]


class Rock(RayBased):
    def __init__(self, location: Square, piece: str):
        super().__init__(location, piece)
        row, col = location
        self.rays = [
            [(row, c) for c in range(col + 1, 8)],
            [(row, c) for c in range(col - 1, -1, -1)],
            [(r, col) for r in range(row + 1, 8)],
            [(r, col) for r in range(row - 1, -1, -1)],
        ]


class Bishop(RayBased):
    def __init__(self, location: Square, piece: str):
        super().__init__(location, piece)
        row, col = location
        self.rays = [
            [(r, c) for r, c in zip(range(row + 1, 8), range(col + 1, 8))],
            [(r, c) for r, c in zip(range(row - 1, -1, -1), range(col + 1, 8))],
            [(r, c) for r, c in zip(range(row + 1, 8), range(col - 1, -1, -1))],
            [(r, c) for r, c in zip(range(row - 1, -1, -1), range(col - 1, -1, -1))],
        ]


class Knight(Piece):
    def __init__(self, location: Square, piece: str):
        super().__init__(location, piece)
        row, col = location
        self.moves = []
        for add_row in [1, 2]:
            add_col = 3 - add_row
            for sign_row in [-1, 1]:
                r = row + sign_row * add_row
                if r > 7 or r < 0:
                    continue
                for sign_col in [-1, 1]:
                    c = col + sign_col * add_col
                    if c > 7 or c < 0:
                        continue
                    self.moves.append((r, c))

    def theoretical_moves(
            self, get_square_status: GetSquareStatus) -> Generator[Square, None, None]:
        for move in self.moves:
            status = get_square_status(move) 
            if (
                status == ' ' # free
                or status.islower() != self.piece.islower() # capture
            ):
                yield move


class King(Piece):
    def __init__(self, location: Square, piece: str):
        super().__init__(location, piece)
        row, col = location
        self.moves = []
        for add_row in [-1, 0, 1]:
            r = row + add_row
            if r < 0 or r > 7:
                continue
            for add_col in [-1, 0, 1]:
                if add_row == 0 and add_col == 0:
                    continue
                c = col + add_col
                if c < 0 or c > 7:
                    continue
                self.moves.append((r, c))

    def theoretical_moves(
            self, get_square_status: GetSquareStatus) -> Generator[Square, None, None]:
        for move in self.moves:
            status = get_square_status(move) 
            if (
                status == ' ' # free
                or status.islower() != self.piece.islower() # capture
            ):
                yield move


class Game:
    """Representing a current state or a Chess game.
    A lot was copied from Chessnut, yet I wanted to speed up exploration and evaluation.
    """
    pass


@dataclass
class Board:
    turn: str # should be either 'w' or 'b'
    casteling: str # options still available
    en_passant: str
    half_moves: int
    move_number: int
    stats: Dict[str, int] # how many of each piece are there. There might be some extra stuff there
    piece_placement: 'np.array'
    location_to_piece: Dict[Square, Piece]

    def simple_heuristic(self) -> Tuple[float, float]:
        vals = {
            'p': 1.0,
            'n': 3.0,
            'b': 3.0,
            'r': 5.0,
            'q': 8.0,
        }
        white, black = 0., 0.
        for piece, count in self.stats.items():
            if piece.islower():
                white += vals.get(piece, 0) * count
            else:
                black += vals.get(piece.lower(), 0) * count
        return white, black


default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Game:
    def __init__(self,
        turn: str,
        casteling: str,
        en_passant: str,
        half_moves: int,
        move_number: int,
        stats: Dict[str, int],
        piece_placement: 'np.array',
        location_to_piece: Dict[Square, Piece]
        ):
        self.board = Board(
            turn=turn,
            casteling=casteling,
            en_passant=en_passant,
            half_moves=int(half_moves),
            move_number=int(move_number),
            stats=stats,
            piece_placement=piece_placement,
            location_to_piece=location_to_piece
        )

    @classmethod
    def from_fen(cls, fen: str = default_fen):
        (
            position,
            turn,
            casteling,
            en_passant,
            half_moves,
            move_number
        ) = fen.split(' ')

        def split_and_expand(line: str) -> Generator[str, None, None]:
            for char in line:
                if char.isdigit():
                    for _ in range(int(char)):
                        yield ' '
                else:
                    yield char

        piece_placement = np.array([
            list(split_and_expand(line))
            for line in position.split('/')
        ])

        return cls(
            turn=turn,
            casteling=casteling,
            en_passant=en_passant,
            half_moves=int(half_moves),
            move_number=int(move_number),
            stats=Counter(position),
            piece_placement=piece_placement,
            location_to_piece={
                (row, col): cls.piece_for(piece_placement[row, col], row, col)
                for row in range(8)
                for col in range(8)
                if piece_placement[row, col] != ' '
            }
        )

    @classmethod
    def piece_for(cls, piece: str, row: int, col: int):
        match piece:
            case 'p' | 'P':
                return Pawn((row, col), piece)
            case 'r' | 'R':
                return Rock((row, col), piece)
            case 'n' | 'N':
                return Knight((row, col), piece)
            case 'b' | 'B':
                return Bishop((row, col), piece)
            case 'q' | 'Q':
                return Queen((row, col), piece)
            case 'k' | 'K':
                return King((row, col), piece)

    def available_moves(self) -> Generator[Tuple[str, Game], None, None]:
        other = "w" if self.board.turn == "b" else "b"
        next_move_number = self.board.move_number if other == "b" else self.board.move_number + 1
        for move in self._raw_moves():
            stats = self.board.stats.copy()
            piece_placement = np.copy(self.board.piece_placement)
            location_to_piece = self.board.location_to_piece.copy()
            row_from, col_from, row_to, col_to, piece = move
            status_dest = self.board.piece_placement[row_to, col_to]
            if status_dest != ' ':
                stats[status_dest] -= 1
            location_to_piece[row_to, col_to] = Game.piece_for(piece, row_to, col_to) # potentially override
            status_src = self.board.piece_placement[row_from, col_from]
            del location_to_piece[row_from, col_from] # the piece is already in its new place
            piece_placement[row_from, col_from] = ' '
            piece_placement[row_to, col_to] = status_src
            g = Game(
                other,
                self.board.casteling,
                self.board.en_passant,
                self.board.half_moves,
                next_move_number,
                stats,
                piece_placement,
                location_to_piece,
            )
            yield f'{move}', g

    def _raw_moves(self) -> Generator[Tuple[int, int, int, int, str], None, None]:
        for r, c, p in self.__current_player_pieces():
            piece = self.board.location_to_piece[r, c]
            for dest in piece.theoretical_moves(lambda square: self.board.piece_placement[square]):
                yield r, c, *dest, p

    def __current_player_pieces(self) -> Generator[Tuple[int, int, str], None, None]:
        relevant_pieces = (
            ['p', 'r', 'n', 'b', 'q', 'k']
            if self.board.turn == 'w'
            else ['P', 'R', 'N', 'B', 'Q', 'K']
        )
        for r, c in zip(*np.where(np.isin(self.board.piece_placement, relevant_pieces))):
            yield r.item(), c.item(), self.board.piece_placement[r, c].item()

    def __repr__(self):
        return f"""{self.board}"""
