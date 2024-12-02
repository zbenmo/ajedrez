from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
from typing import Dict, Generator, List, Tuple, Callable
from collections import Counter


Square = Tuple[int, int]
GetSquareStatus = Callable[[Square], str]

W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "


class Matrix:
    """A simple replacement for numpy matrix."""

class Matrix:
    def __init__(self, values: List[List[str]]):
        self.values = values

    def __getitem__(self, square: Square) -> str:
        (r, c) = square
        return self.values[r][c]

    def __setitem__(self, square: Square, val) -> Matrix:
        (r, c) = square
        new_row = self.values[r][:]
        new_row[c] = val
        self.values[r] = new_row
        return self

    def find(self, val: str) -> Tuple[int, int]:
        for r_i, row in enumerate(self.values):
            if val in row:
                return (r_i, row.index(val))

    def copy(self) -> Matrix:
        return Matrix(self.values[:]) # a real copy will be only on write

    def where(self, vals) -> Generator[Tuple[int, int, str], None, None]:
        for r_i, row in enumerate(self.values):
            for c_i, val in enumerate(row):
                if val in vals:
                    yield r_i, c_i, val


class Piece(ABC):
    """"""
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
    """"""
    def __init__(self, location: Square, piece: str):
        super().__init__(location, piece)
        direction = +1 if piece == W_P else -1
        starting_row = 1 if piece == W_P else 6
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
            if get_square_status(move) == EMPTY:
                yield move
            else:
                break
        for move in self.captures:
            status = get_square_status(move)
            if status != EMPTY and status.islower() != self.piece.islower():
                yield move


class RayBased(Piece, ABC):
    """"""
    def __init__(self, location, piece):
        super().__init__(location, piece)

    def theoretical_moves(
            self, get_square_status: GetSquareStatus) -> Generator[Square, None, None]:
        for ray in self.rays:
            for move in ray:
                status = get_square_status(move)
                if status == EMPTY:
                    yield move
                else:
                    if status.islower() != self.piece.islower():
                        yield move # capture
                    break # this ray reached a piece


class Queen(RayBased):
    """"""
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


class Rook(RayBased):
    """"""
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
    """"""
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
    """"""
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
                status == EMPTY
                or status.islower() != self.piece.islower() # capture
            ):
                yield move


class King(Piece):
    """"""
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
                status == EMPTY
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
    castling_rights: str # options still available
    en_passant: str
    half_moves: int
    move_number: int
    stats: Dict[str, int] # how many of each piece are there. There might be some extra stuff there
    piece_placement: Matrix # 'np.array'
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
                black += vals.get(piece, 0) * count
            else:
                white += vals.get(piece.lower(), 0) * count
        return white, black


default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Game:
    def __init__(self,
        turn: str,
        castling_rights: str,
        en_passant: str,
        half_moves: int,
        move_number: int,
        stats: Dict[str, int],
        piece_placement: Matrix, #'np.array',
        location_to_piece: Dict[Square, Piece]
        ):
        self.board = Board(
            turn=turn,
            castling_rights=castling_rights,
            en_passant=en_passant,
            half_moves=half_moves,
            move_number=move_number,
            stats=stats,
            piece_placement=piece_placement,
            location_to_piece=location_to_piece
        )

    def _verify(self):
        for r_i, row in enumerate(self.board.piece_placement.values):
            for c_i, val in enumerate(row):
                if val != EMPTY:
                    assert self.board.location_to_piece[r_i, c_i].piece == val, f'{r_i=}, {c_i=}, {val=}'
        # TODO: check more

    def to_fen(self) -> str:

        def compress_and_join(line: List[str]) -> str:
            pieces = []
            count_empty = 0
            for piece in line:
                if piece != ' ':
                    if count_empty > 0:
                        pieces.append(str(count_empty))
                        count_empty = 0
                    pieces.append(piece)
                else:
                    count_empty += 1
            if count_empty > 0:
                pieces.append(str(count_empty))
            return ''.join(pieces)

        position = '/'.join(
            compress_and_join(line) for line in 
            reversed(self.board.piece_placement.values)
        )
        return f'{position} {self.board.turn} {self.board.castling_rights} {self.board.en_passant} {self.board.half_moves} {self.board.move_number}'

    @classmethod
    def from_fen(cls, fen: str = default_fen):
        (
            position,
            turn,
            castling_rights,
            en_passant,
            half_moves,
            move_number
        ) = fen.split(' ')

        def split_and_expand(line: str) -> Generator[str, None, None]:
            for char in line:
                if char.isdigit():
                    for _ in range(int(char)):
                        yield EMPTY
                else:
                    yield char

        piece_placement = Matrix([
            list(split_and_expand(line))
            for line in reversed(position.split('/'))
        ])

        return cls(
            turn=turn,
            castling_rights=castling_rights,
            en_passant=en_passant,
            half_moves=int(half_moves),
            move_number=int(move_number),
            stats=Counter(position),
            piece_placement=piece_placement,
            location_to_piece={
                (row, col): cls.piece_for(piece_placement[row, col], row, col)
                for row in range(8)
                for col in range(8)
                if piece_placement[row, col] != EMPTY
            }
        )

    @classmethod
    def piece_for(cls, piece: str, row: int, col: int):
        match piece:
            case 'p' | 'P':
                return Pawn((row, col), piece)
            case 'r' | 'R':
                return Rook((row, col), piece)
            case 'n' | 'N':
                return Knight((row, col), piece)
            case 'b' | 'B':
                return Bishop((row, col), piece)
            case 'q' | 'Q':
                return Queen((row, col), piece)
            case 'k' | 'K':
                return King((row, col), piece)

    def make_specific_move(self, move_to_take: Tuple[int, int, int, int, str, str]) -> Game | None:
        for move in self._available_moves():
            if move != move_to_take:
                continue
            g = self._verify_move(move)
            if g:
                return g
        return None

    def _available_moves(self) -> Generator[Tuple[int, int, int, int, str, str], None, None]:
        yield from itertools.chain(
            self._raw_moves(player=self.board.turn), self._castling_raw_moves(player=self.board.turn))

    def _verify_move(self, move: Tuple[int, int, int, int, str, str]) -> Game:
        other = "w" if self.board.turn == "b" else "b"
        next_move_number = self.board.move_number if other == "b" else self.board.move_number + 1
        stats = self.board.stats.copy()
        piece_placement = self.board.piece_placement.copy()
        location_to_piece = self.board.location_to_piece.copy()
        castling_rights = self.board.castling_rights
        half_moves = self.board.half_moves

        half_moves += 1 # till shown otherwise

        row_from, col_from, row_to, col_to, piece, promotion = move

        if piece == W_P or piece == B_P:
            half_moves = 0 # a Pawn made a move

        status_dest = self.board.piece_placement[row_to, col_to]
        if status_dest != EMPTY:
            stats[status_dest] -= 1
            half_moves = 0 # a capture
            # Rook was eaten - potentially before attempt to castling.
            if (status_dest == W_R and row_to == 0) or (status_dest == B_R and row_to == 7):
                if col_to == 0:
                    castling_rights = castling_rights.replace('Q' if status_dest == W_R else 'q', '')
                elif col_to == 7:
                    castling_rights = castling_rights.replace('K' if status_dest == W_R else 'k', '')
        location_to_piece[row_to, col_to] = (
            Game.piece_for(promotion or piece, row_to, col_to) # potentially override
        )
        del location_to_piece[row_from, col_from] # the piece is already in its new place
        piece_placement[row_from, col_from] = EMPTY
        piece_placement[row_to, col_to] = promotion or piece
        if promotion:
            stats[piece] -= 1
            stats[promotion] += 1

        # en-passant (capture)
        if (status_dest == EMPTY) and (piece == W_P or piece == B_P) and (col_from != col_to):
            target_row = row_to - 1 if piece == W_P else row_to + 1
            status_dest = piece_placement[target_row, col_to]
            piece_placement[target_row, col_to] = EMPTY
            assert status_dest == W_P or status_dest == B_P, f'{target_row=}, {col_from=}, {col_to=}, {status_dest=}'
            stats[status_dest] -= 1
            del location_to_piece[target_row, col_to]

        # castling
        is_castling = False
        if (piece == W_K or piece == B_K) and (abs(col_from - col_to) == 2):
            is_castling = True
            if col_to == 6:
                piece_placement[row_to, 5] = piece_placement[row_to, 7]
                piece_placement[row_to, 7] = EMPTY
                del location_to_piece[row_to, 7]
                location_to_piece[row_to, 5] = Game.piece_for(piece_placement[row_to, 5], row_to, 5)
            elif col_to == 2:
                piece_placement[row_to, 3] = piece_placement[row_to, 0]
                piece_placement[row_to, 0] = EMPTY
                del location_to_piece[row_to, 0]
                location_to_piece[row_to, 3] = Game.piece_for(piece_placement[row_to, 3], row_to, 3)
            else:
                assert False, f'{col_to=}'

        en_passant = '-' # for next turn
        if (
            (piece == W_P and (row_from == 1) and (row_to == 3))
            or
            (piece == B_P and (row_from == 6) and (row_to == 4))
            ):
            en_passant = f"{chr(ord('a') + col_from)}{3 if row_to == 3 else 6}"

        # update castling for next turn
        if piece == W_K:
            castling_rights = castling_rights.replace('K', '')
            castling_rights = castling_rights.replace('Q', '')
        elif piece == B_K:
            castling_rights = castling_rights.replace('k', '')
            castling_rights = castling_rights.replace('q', '')
        elif piece == W_R and row_from == 0:
            if col_from == 0:
                castling_rights = castling_rights.replace('Q', '')
            elif col_from == 7:
                castling_rights = castling_rights.replace('K', '')
        elif piece == B_R and row_from == 7:
            if col_from == 0:
                castling_rights = castling_rights.replace('q', '')
            elif col_from == 7:
                castling_rights = castling_rights.replace('k', '')
        if len(castling_rights) < 1:
            castling_rights = '-'

        g = Game(
            other,
            castling_rights,
            en_passant,
            half_moves,
            next_move_number,
            stats,
            piece_placement,
            location_to_piece,
        )

        # g._verify()

        # is it a valid game/board?
        if g._is_check(which_king=self.board.turn):
            assert not is_castling # because I believe I've already verified there
            # we either did not protect the king, or have exposed it
            return None # it is not, that move should not be considered

        return g

    def available_moves(self) -> Generator[Tuple[str, Game], None, None]:
        for move in self._available_moves():

            g = self._verify_move(move)

            row_from, col_from, row_to, col_to, piece, promotion = move
            move_formated = (
                f'{chr(ord("a") + col_from)}{row_from + 1}{chr(ord("a") + col_to)}{row_to + 1}{promotion or ""}'
            )

            yield move_formated, g

    def is_check(self) -> bool:
        return self._is_check(self.board.turn)

    def _is_check(self, which_king: str) -> bool:
        """Returns wheater the relevant king is being treated."""
        king = W_K if which_king == "w" else B_K
        other = "b" if which_king == "w" else "w"
        king_location = self.board.piece_placement.find(king)
        return self._is_treatened_by(king_location, by=other)

    def _is_treatened_by(self, square: Square, by: str):
        """Returns wheater the relevant squares is being tretened.
        'by' should be either 'w' or 'b'
        """
        for move in self._raw_moves(player=by):
            _, _, row_to, col_to, _, _ = move
            if square == (row_to, col_to):
                return True
        return False

    def _raw_moves(self, player: str) -> Generator[Tuple[int, int, int, int, str, str], None, None]:
        for r, c, p in self._player_pieces(player):
            piece: Piece = self.board.location_to_piece[r, c]

            # a bit logic for en-passant, if relevant, pretend you have there a Pawn
            en_passant_square = None # for en-passant
            if self.board.en_passant != '-' and isinstance(piece, Pawn):
                # above isinstance(piece, Pawn) is redundant yet added for clarity
                en_passant = ord(self.board.en_passant[0]) - ord('a')
                if c != en_passant: # if it is exactly in the current column, we don't want to "add" it
                    if p == W_P:
                        en_passant_square = (5, en_passant)
                        en_passant_piece = B_P
                    elif p == B_P:
                        en_passant_square = (2, en_passant)
                        en_passant_piece = W_P

            for dest in piece.theoretical_moves(
                lambda square: en_passant_piece if square == en_passant_square else self.board.piece_placement[square]
                ):
                # offering options for promotion is done here, if relevant
                if p == W_P and dest[0] == 7:
                    for promotion in [W_Q, W_R, W_B, W_N]:
                        yield r, c, *dest, p, promotion
                elif p == B_P and dest[0] == 0:
                    for promotion in [B_Q, B_R, B_B, B_N]:
                        yield r, c, *dest, p, promotion
                else:
                    yield r, c, *dest, p, None

    def _castling_raw_moves(self, player: str) -> Generator[Tuple[int, int, int, int, str, str], None, None]:
        assert player == self.board.turn
        if player == "w":
            other = "b"
            row = 0
            king_castling = "K"
            queen_castling = "Q"
            king = W_K
            rook = W_R
        else:
            other = "w"
            row = 7
            king_castling = "k"
            queen_castling = "q"
            king = B_K
            rook = B_R

        if (
            (king_castling not in self.board.castling_rights)
            and
            (queen_castling not in self.board.castling_rights)
        ):
            return

        if self.is_check():
            return

        assert self.board.piece_placement[row, 4] == king

        if king_castling in self.board.castling_rights:
            assert self.board.piece_placement[row, 7] == rook
            if self.board.piece_placement[row, 5] == EMPTY and self.board.piece_placement[row, 6] == EMPTY:
                if not self._is_treatened_by((row, 5), other) and not self._is_treatened_by((row, 6), other):
                    yield row, 4, row, 6, king, None

        if queen_castling in self.board.castling_rights:
            assert self.board.piece_placement[row, 0] == rook
            if (
                self.board.piece_placement[row, 3] == EMPTY
                and self.board.piece_placement[row, 2] == EMPTY
                and self.board.piece_placement[row, 1] == EMPTY
            ):
                if (
                    not self._is_treatened_by((row, 3), other) and not self._is_treatened_by((row, 2), other)
                ):
                    yield row, 4, row, 2, king, None

    def _player_pieces(self, player: str) -> Generator[Tuple[int, int, str], None, None]:
        relevant_pieces = (
            [W_P, W_R, W_N, W_B, W_Q, W_K]
            if player == 'w'
            else [B_P, B_R, B_N, B_B, B_Q, B_K]
        )
        yield from self.board.piece_placement.where(relevant_pieces)

    def __repr__(self):
        return f"""{self.board}"""
