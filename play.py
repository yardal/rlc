import chess
import random
import time
import logging
import random
import os
import numpy as np


max_depth = 2

from search import SearchNode, alpha_beta_search
from cnn import Network
from fen import get_tensor_from_FEN

def get_tree(board):
    root = SearchNode(board.board_fen(), -1)
    return get_all_moves(root, board, 0, max_depth)


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


def get_score(board):
    score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    return score[board.result()]


def get_minimax_move(board, tree, is_white, net):

    for move in tree.children:
        move.val = alpha_beta_search(move, max_depth-2, -5.0, 5.0, not is_white, net)

    best_fen = ""
    if is_white:
        best = -1.0
        for child in tree.children:
            if best < child.val:
                best = child.val
                best_fen = child.data

    else:
        best = 1.0
        for child in tree.children:
            if best > child.val:
                best = child.val
                best_fen = child.data


    for i in board.legal_moves:
        board.push(i)
        if board.fen() == best_fen:
            board.pop()
            return i
        board.pop()

    raise Exception("Error! no move found")


def play(board, net, is_white):
    tree = get_tree(board)
    move = get_minimax_move(board, tree,  is_white, net)

    return move


def random_move(board):
    moves = []
    for move in board.legal_moves:
        moves.append(move)

    idx = random.randint(1, len(moves)) - 1
    return moves[idx]


def self_play(net):
    board = chess.Board()
    data_dir = "data"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    game_counter = 0
    self_play_prob = 0.5

    for i in range(100000):
        board.reset()
        move_counter = 0

        while not board.is_game_over():
            is_white = move_counter % 2 == 0
            move_counter = move_counter + 1
            prob = random.uniform(0, 1)
            if prob > self_play_prob:
                move = play(board, net, is_white)
            else:
                move = random_move(board)
            board.push(move)
        game_counter = game_counter + 1
        result = get_score(board)

        file_name = os.path.join(data_dir, str(game_counter) + "_" + str(result))
        np.save(file_name, get_tensor_from_FEN(board.fen()))


def test(a, b):
    white = Network(a)
    black = Network(b)
    print("training")
    white.train()

    print("playing")
    board = chess.Board()
    for i in range(10):
        print(i)
        board.reset()
        while not board.is_game_over():
            move = play(board, white, True)
            if  board.is_game_over():
                break
            board.push(move)
            move = play(board, black, False)
            board.push(move)
        print (board.result())

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.basicConfig(filename='log.txt', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    net = Network("model")
    self_play(net)
    # net.train()
    #test("model", "model2")


if __name__ == "__main__":
    main()
