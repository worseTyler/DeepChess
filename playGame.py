from keras import models, layers, Model, Input, losses
import chess.pgn
import numpy as np
from convert import *

deepChess =  models.load_model('./models/deepChess_possibleDemo.h5')

def get_computer_move(board):
    copy_board = board.copy() 
    move = miniMax(copy_board, 1, True)
    board.push(move)

def getBestMove(board, isMaximizingPlayer):
    legal_moves = []
    for move in board.legal_moves:
        copy_board = board.copy()
        copy_board.push(move)

        if copy_board.is_game_over():
            # found checkmate
            
            # if maximizing player want to always take checkmate
            if isMaximizingPlayer:
                return move
            
            # if minimizing player want to avoid checkmate
            continue

        legal_moves.append(move)

    if len(legal_moves) == 0:
        # forced mate
        return None

    if len(legal_moves) == 1:
        return legal_moves[0]

    bestMove = legal_moves[0] 
    for move in legal_moves[:1]:
        bestMove = compareTwoMoves(board, bestMove, move, isMaximizingPlayer)

    return bestMove

def compareTwoMoves(board, firstMove, secondMove, isMaximizingPlayer):    
    if firstMove is None:
        return secondMove
    if secondMove is None:
        return firstMove
    
    bit_strings = []
    legal_moves = []
    for move in [firstMove, secondMove]:
        legal_moves.append(move)

        copy_board = board.copy()
        copy_board.push(move)
        bit_strings.append(convertBoardToBits(copy_board).reshape(-1,773))
 
    outcome = deepChess.predict([bit_strings[0], bit_strings[1]], verbose=0)
    
    if isMaximizingPlayer:
        if np.argmax(outcome[0]) == 1:
            return secondMove
        return firstMove
    else: 
        if np.argmin(outcome[0]) == 1:
            return secondMove
        return firstMove


# Used this video as reference
# https://www.youtube.com/watch?v=l-hh51ncgDI
def miniMax(board, depth, isMaximizingPlayer):
    if depth <= 0:
        return getBestMove(board, isMaximizingPlayer)
    
    bestMove = None
    bestWorstMove = None

    for move in board.legal_moves:
        copy_board = board.copy()
        copy_board.push(move)
        currentWorstMove = miniMax(copy_board, depth - 1, not isMaximizingPlayer)
        if bestMove is None and bestWorstMove is None:
            bestMove = move
            bestWorstMove = currentWorstMove
        else:
            if compareTwoMoves(board, bestWorstMove, currentWorstMove, isMaximizingPlayer) is not bestWorstMove:
                bestWorstMove = currentWorstMove
                bestMove = move
    
    return bestMove

def start_game():
    board = chess.Board()

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Qh5")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Nf6")
    # print(board)
    # move = chess.Move(chess.H5,chess.F7)
    # board.push(move)
    # print(board.is_capture(move))
    # print(board.is_game_over())
    while not board.is_game_over():
        get_computer_move(board)
        print(board)
        # input("Press enter to continue...")

start_game()


