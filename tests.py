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

def test_4():
    game = Game.from_fen('rnb1k1nr/pppp1pp1/5q2/7p/2N5/b7/PPPPPPPP/R1BQKBNR w KQkq - 3 5')
    list(game.available_moves())

def test_5():
    game = Game.from_fen('rnbqkbnr/2p1pppp/8/1N1p4/p7/8/PPPPPPPP/R1BQKBNR w Kkq - 0 5')
    list(game.available_moves())

def test_6():
    game = Game.from_fen('r3kbn1/1p6/n3pPbr/1N1p3p/2p5/2P5/PP1PPPP1/R1B1KBNR w - - 0 23')
    # list(game.available_moves())
    moves_to_game = dict(game.available_moves())

    # moves = list(moves_to_game)

    def get_value(game):
        me, they = game.board.simple_heuristic()
        return me - they
            
    stat = {}
    for m, g in moves_to_game.items():
        try:
            counts = []
            for m_o, g_o in g.available_moves():
                counts.append(get_value(g_o))
            stat[m] = min(counts)
        except:
            print(f"{m=}")
            raise
    assert len(max(stat, key=stat.get)) > 0 # ?
