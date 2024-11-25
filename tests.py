from .ajedrez import Game


def test_en_passant():
    game = Game.from_fen('rnbqkbnr/p2ppp1p/8/1pp3p1/6P1/2P2P2/PP1PP2P/RNBQKBNR w KQkq g6 0 4')
    list(game.available_moves())


def test_2():
    """was something with Pawn jumping twice, and also maybe a missing knight move"""
    game = Game.from_fen('2b2r2/3p2pp/2n1k1q1/r1Pp1p2/p1p1P3/6PQ/P1PBN2P/2RK3R w - - 4 25')
    list(game.available_moves())


def test_3():
    """bug in the code related to casteling"""
    game = Game.from_fen('r2qkbnr/p1pppppp/b7/1B6/1Q6/4P3/PPPP1PPP/RNB1K1NR w KQkq - 1 5')
    list(game.available_moves())
