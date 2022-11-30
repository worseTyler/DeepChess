import chess.pgn
import numpy as np
import time

emptySixBits = "000000"
positionToBits = {
    # black pieces
    "p" : "100000" + emptySixBits,
    "b" : "010000" + emptySixBits,
    "n" : "001000" + emptySixBits,
    "r" : "000100" + emptySixBits,
    "q" : "000010" + emptySixBits,
    "k" : "000001" + emptySixBits,
    
    # white pieces
    "P" : emptySixBits + "100000",
    "B" : emptySixBits + "010000",
    "N" : emptySixBits + "001000",
    "R" : emptySixBits + "000100",
    "Q" : emptySixBits + "000010",
    "K" : emptySixBits + "000001",

    "empty" : emptySixBits + emptySixBits
}

def convertBoardToBits(board):
    # rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
    # Fen for starting board
    fen = board.board_fen()
    fen_segments = fen.split("/")

    bit_string = ""
    for segment in fen_segments:
        for piece in segment:
            if piece.isdigit():
                bit_string += positionToBits["empty"] * int(piece)
            else:
                bit_string += positionToBits[piece]

    bit_string += f"{int(board.turn)}"
    # Black castling rights
    bit_string += f"{int(board.has_kingside_castling_rights(0))}"
    bit_string += f"{int(board.has_queenside_castling_rights(0))}"
    # White castling rights
    bit_string += f"{int(board.has_kingside_castling_rights(1))}"
    bit_string += f"{int(board.has_queenside_castling_rights(1))}"

    # print(fen_segments)
    # if len(bit_string) != 773:
    #     print("BAD BAD")
    #     exit(1)
    
    return np.fromstring(' '.join(bit_string), dtype=int, sep=" ")

# pgn = open("./chessGames/someGames.pgn")
# games = []
# start = time.perf_counter()
# game = chess.pgn.read_game(pgn)
# for i in range(3000):
# # while game is not None:
#     result_string = game.headers["Result"]
#     if result_string != "1/2-1/2":
#         games.append(game)
#     game = chess.pgn.read_game(pgn)
# end = time.perf_counter()
# print(f"Time: {end - start} seconds")

bit_strings = []
labels = []

pgn = open("./chessGames/allGames.pgn")
game = chess.pgn.read_game(pgn)

start = time.perf_counter()
#for i in range(100):
# for game in games:
while game is not None:
    result_string = game.headers["Result"]
    if result_string is None or result_string == "1/2-1/2":
        # game ended in a draw
        # dont want to include this in dataset
        game = chess.pgn.read_game(pgn)
        continue
    
    # 1 if white wins, 0 if black wins
    result = 1 if result_string == "1-0" else 0

    board = game.board() 
    move_counter = 0
    for move in game.mainline_moves():
        move_counter += 1
        # have to check if move is capture before pushing
        if board.is_capture(move):
            # if the move is a capture, skip
            board.push(move)
            continue

        board.push(move)
        if move_counter < 5:
            # if move is in first five of game, skip
            continue
        
        bit_string = convertBoardToBits(board)
        bit_strings.append(bit_string)
        labels.append(result)
    game = chess.pgn.read_game(pgn)

end = time.perf_counter()
print(f"Time: {end - start} seconds")

data = np.array(bit_strings)
labels = np.array(labels)

print(data.shape)
print(labels.shape)

np.save("./data/bit_strings_15.npy", data)
np.save("./data/labels_15.npy", labels)