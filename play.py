import chess
import random
import time
import logging
import random
import os
import numpy as np
import tensorflow as tf


max_depth = 2

from search import SearchNode, alpha_beta_search
from cnn import Network


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


def self_play(board, net):
    self_play = 0.5
    draw_result = "1/2-1/2"
    result = draw_result
    while result == draw_result:
        board.reset()
        move_counter = 0
        while not board.is_game_over():
            is_white = move_counter % 2 == 0
            move_counter = move_counter + 1
            prob = random.uniform(0, 1)
            if prob > self_play:
                move = play(board, net, is_white)
            else:
                move = random_move(board)
            board.push(move)
        print(board)
        print(board.result())
        print(board.fen())
        print()
        print()
        print()
        result = board.result()


def main():
    board = chess.Board()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.basicConfig(filename='log.txt', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    net = Network()

    return self_play(board, net)


if __name__ == "__main__":
    main()
