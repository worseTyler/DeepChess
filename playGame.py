from keras import models, layers, Model, Input, losses
import chess.pgn
import numpy as np
from util import *

deepChess = models.load_model('./models/deepChess')

def get_computer_move(board):
    legal_moves = list(map(lambda x: x, board.legal_moves))
    bit_strings = []
    for move in legal_moves:
        copy_board = board.copy()
        copy_board.push(move)
        bit_strings.append(convertBoardToBits(copy_board).reshape(-1,773))

    best_bit_string_index = 0
    for i, string in enumerate(bit_strings[1:]):
        outcome = deepChess.predict([bit_strings[best_bit_string_index], string], verbose=0)
        if np.argmax(outcome[0]) == 1:
            best_bit_string_index = i

    board.push(legal_moves[best_bit_string_index])

def player_move(board):
    print(board.legal_moves)
    moves = board.legal_moves
    moves_list = []
    print("Select One of the following moves: ")
    for i, move in enumerate(moves):
        print(f"{i + 1}: {move}")
        moves_list.append(move)
        
    move = int(input(">>> "))
    board.push(moves_list[move-1])

def start_game():
    board = chess.Board()

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Qh5")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Nf6")
    print(board)
    for i in range(50):
        if i % 2 == 0:
            get_computer_move(board)
        else:
            get_computer_move(board)
        
        print(board)
        input("Press enter to continue...")

start_game()


