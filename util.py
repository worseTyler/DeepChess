import chess.pgn
import numpy as np

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