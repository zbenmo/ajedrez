from .ajedrez import Game


def test_en_passant():
    game = Game.from_fen('rnbqkbnr/p2ppp1p/8/1pp3p1/6P1/2P2P2/PP1PP2P/RNBQKBNR w KQkq g6 0 4')
    list(game.available_moves())
