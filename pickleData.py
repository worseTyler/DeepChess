import numpy as np
import chess.pgn
import pickle
import time
import sys


sys.setrecursionlimit(10000)

def pickle_helper(total_game_counter, white_win_counter, white_loss_counter, white_wins, white_losses):
    with open(f"./pickles/white_wins_{total_game_counter}_{white_win_counter}", "wb") as f:
        pickle.dump(white_wins, f)

    with open(f"./pickles/white_losses_{total_game_counter}_{white_loss_counter}", "wb") as f:
        pickle.dump(white_losses, f)


pgn = open("./chessGames/allGames.pgn")
game = chess.pgn.read_game(pgn)
game_counter = 1

white_win_counter = 0
draw_counter = 0
black_win_counter = 0

white_wins = []
white_losses = []

total_start = time.perf_counter()
loop_start = time.perf_counter()
while game is not None:
    game_counter += 1
    if game_counter % 5000 == 0:
        loop_end = time.perf_counter()
        print(f"Time for Loop: {loop_end - loop_start} seconds")
        loop_start = time.perf_counter()
        print(f"{game_counter}: {draw_counter} {white_win_counter} {black_win_counter}")

    if game_counter % 100000 == 0:
        pickle_start = time.perf_counter()
        print(f"Pickling for {game_counter}")
        pickle_helper(game_counter, white_win_counter, black_win_counter, white_wins, white_losses)
        white_wins = []
        white_losses = []
        pickle_end = time.perf_counter()
        print(f"Pickle End: {pickle_end - pickle_start} seconds")

    result_string = game.headers["Result"]
    if result_string is None:
        print(f"Error None String: {game_counter}")
    elif result_string == "1/2-1/2":
        draw_counter += 1
    elif result_string == "1-0":
        white_win_counter += 1
        white_wins.append(game)
    elif result_string == "0-1":
        black_win_counter += 1
        white_losses.append(game)
    else:
        print(f"Error Unknown Result: {result_string}")

    game = chess.pgn.read_game(pgn)

print(f"Draw Counter: {draw_counter}")
print(f"White Win Counter: {white_win_counter}")
print(f"Black Win Counter: {black_win_counter}")
print(f"Total Games: {game_counter}")
print(f"Total Games Finished: {white_win_counter + black_win_counter}")

pickle_helper(game_counter, white_win_counter, black_win_counter, white_wins, white_losses)

total_end = time.perf_counter()
print(f"Total Time: {total_end - total_start} seconds")