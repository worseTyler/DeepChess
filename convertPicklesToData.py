import pickle
import time
import tables as tb
import chess.pgn
import random
from convert import *



def unpickle(filename):
    st = time.perf_counter()
    with open("./pickles/" + filename, "rb") as f:
        unpickled = (pickle.load(f))
    e = time.perf_counter()
    print(f"Pickle Time: {e - st}")
    return unpickled

def generateTable(files, table_name):
    s = time.perf_counter()
    i = 0

    table_file = f"./data/{table_name}.h5"
    fileh = tb.open_file(table_file, mode='w')    
    dataTable = None

    for file in files:
        print(f"Starting on {file}")
        games = unpickle(file)
        start_game = time.perf_counter()

        bit_strings = []
        for game in games:
            board = game.board()
            i += 1
            move_counter = 0
            single_game_bit_strings = []
            for move in game.mainline_moves():
                move_counter += 1
                # have to check if move is capture before pushing
                if board.is_capture(move) or move_counter < 5:
                    # if the move is a capture, skip
                    # if move is in first five of game, skip
                    
                    # if move is pushed before checking capture, it always returns true
                    board.push(move)
                    continue

                board.push(move)
                single_game_bit_strings.append(convertBoardToBits(board))

            # pick a random number of positions from a game
            for i in range(random.randint(8,12)):
                if len(single_game_bit_strings) == 0:
                    break
                
                # sample without replacement
                random_bit_string = random.choice(single_game_bit_strings)
                single_game_bit_strings = list(filter(lambda x: not np.array_equal(random_bit_string, x), single_game_bit_strings))
                bit_strings.append(random_bit_string)

            if i % 1000 == 0:
                print(f"Gone through {i} games")

        print(f"Number of Positions: {len(bit_strings)}")
        bit_strings = np.array(bit_strings)
        
        table_start = time.perf_counter()
        if dataTable is None:
            dataTable = fileh.create_earray(fileh.root, "bitStrings", obj=bit_strings)
        else:
            dataTable.append(bit_strings)
        table_end = time.perf_counter()
        print(f"Table Time: {table_end - table_start} seconds")

        end_game = time.perf_counter()
        print(f"Game: {end_game - start_game} seconds")

    e = time.perf_counter()
    print(f"Time: {e-s}")

# convert these files to the pickles you created
win_files = [
    "white_wins_100000_36075", 
    "white_wins_200000_72100", 
    "white_wins_300000_107597", 
    "white_wins_400000_142914", 
    "white_wins_500000_175815", 
    "white_wins_600000_208780",
    "white_wins_700000_241090",
    "white_wins_800000_274118",
    "white_wins_900000_310393"
]

loss_files = [
    "white_losses_100000_27424",
    "white_losses_200000_53717",
    "white_losses_300000_80471",
    "white_losses_400000_106489",
    "white_losses_500000_129629",
    "white_losses_600000_154782",
    "white_losses_700000_179102",
    "white_losses_800000_202932",
    "white_losses_900000_229840",
    "white_losses_1000000_257372",
    "white_losses_1100000_282711",
    "white_losses_1200000_303819"
]

generateTable(win_files, "winTable")
generateTable(loss_files, "lossTable")