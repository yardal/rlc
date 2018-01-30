import chess
import random
import time
from cnn import cnn

def random_play():
        
    done = False
    best_i = 10000
    board = chess.Board()
    for i in range(10000):

        moves = []
        for move in board.legal_moves:
            moves.append(move)
            random.shuffle(moves)
        if board.is_checkmate():
            string = ("checkmate")
            done = True
            break
        elif board.is_stalemate():
            string = ("stalemate")
            done = True
            break
        elif board.is_insufficient_material():
            string = ("insuffienct")
            done = True
            break
        else:
            board.push(moves[0])
    if i < best_i:
        best_i = i
        print("best i")
        print(board)
        print(i)
        print(string)
    
def main():
    
   
   
    net = cnn()    
    
    
if __name__ == "__main__":
    main()