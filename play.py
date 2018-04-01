import chess
import random
import time
import logging
import random
import os
import numpy as np
import tensorflow as tf
from cnn import get_network_endpoints
from search import SearchNode


size = 8
max_depth = 2


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


def get_tree(board):
    root = SearchNode(board.board_fen(), -1)
    return get_all_moves(root, board, 0, max_depth)


def get_score(board):
    score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    return score[board.result()]


def evaluate_tree(tree, net):
    leaves = tree.get_leaf_nodes()
    fens = [v.data for v in leaves]
    fens = list(set(fens))

    batch_size = 64
    number_fens = len(fens)
    iterations = int((number_fens + batch_size - 1) / batch_size)

    outputs_ = np.zeros((int(((number_fens + batch_size - 1) / batch_size) * batch_size), 1))

    for i in range(iterations):

        input_ = np.zeros((batch_size, 12, size, size, 1))

        for f in range(batch_size):
            f_index = min(i * batch_size + f, number_fens - 1)
            input_[f] = get_tensor_from_FEN(fens[f_index])

        start_index = i * batch_size
        end_index = (i + 1) * batch_size

        outputs_[start_index:end_index] = net['sess'].run(net['net'], feed_dict={net['x']: input_})

        for f in range(batch_size):
            f_index = min(i * batch_size + f, number_fens - 1)
            board = chess.Board(fens[f_index])
            if board.is_game_over():
                outputs_[f_index] = get_score(board)

    fens_dict = {}
    for i, f in enumerate(fens):
        fens_dict[f] = outputs_[i]

    return fens_dict


def get_minimax_move(board, tree, fens_dict):

    for depth in range(max_depth - 2, -1, -1):
        nodes = tree.get_nodes_by_depth(depth)

        for node in nodes:
            if node.depth + 2 == max_depth:
                #terminal node
                if not node.children:
                    node.val = fens_dict[node.data]
                else:
                    node.val = fens_dict[node.children[0].data]
                    for child in node.children:
                        node.val = min(node.val, fens_dict[child.data])

            else:

                for child in node.children:
                    node.val = node.children[0].val
                    if depth % 2 == max_depth % 2:
                        node.val = min(node.val, child.val)
                    else:
                        node.val = max(node.val, child.val)

    best = tree.children[0].val
    best_fen = tree.children[0].data
    for t in tree.children:
        if best < t.val:
            best = t.val
            best_fen = t.data

    for i in board.legal_moves:
        board.push(i)
        if board.fen() == best_fen:
            board.pop()
            return i
        board.pop()

    raise Exception("Error! no move found")


def play(board, net):

    tree = get_tree(board)
    fens_dict = evaluate_tree(tree, net)
    move = get_minimax_move(board, tree, fens_dict)

    return move


def random_move(board):

    moves = []
    for move in board.legal_moves:
        moves.append(move)

    idx = random.randint(1,len(moves)) - 1
    return moves[idx]


def self_play(board, net):

    self_play = 0.1
    result = "1/2-1/2"
    while result == "1/2-1/2":
        board.reset()
        while not board.is_game_over():
            prob = random.uniform(0, 1)
            if prob > self_play:
                move = play(board, net)
            else:
                move = random_move(board)
            # print(board.fen())
            board.push(move)
            # print()
            # print()
            # print()
        print(board)
        print(board.result())
        print(board.fen())
        print()
        print()
        print()
        result = board.result()

def get_all_moves(tree, board, depth, max_depth):
    if depth == max_depth:
        return tree

    for i in board.legal_moves:
        board.push(i)
        node = SearchNode(board.fen(), depth)
        tree.add_child(node)
        get_all_moves(node, board, depth + 1, max_depth)
        board.pop()
    return tree


def main():
    board = chess.Board()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.basicConfig(filename='log.txt', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    net = get_network_endpoints()

    return self_play(board, net)

if __name__ == "__main__":
    main()
