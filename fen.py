import numpy as np

size = 8

def get_fen_dict():
    return {
        "0": -1,
        "r": 0, "n": 1, "b": 2, "q": 3, "k": 4, "p": 5,
        "R": 6, "N": 7, "B": 8, "Q": 9, "K": 10, "P": 11
    }


def expand_fen(fen):
    new_fen = ""
    for c in fen:
        if c.isdigit():
            new_fen = new_fen + "0" * int(c)
        else:
            new_fen = new_fen + c

    return new_fen


def get_tensor_from_FEN(fen):
    fen = expand_fen(fen)
    fen = fen.replace("/", "")
    board = np.zeros((12, size, size))
    fen_dict = get_fen_dict()
    for i in range(size):
        for j in range(size):
            idx = j + i * size
            value = fen_dict[fen[idx]]
            if (value == -1):
                continue
            board[value][i][j] = 1.0

    return np.expand_dims(np.expand_dims(board, -1), 0)
