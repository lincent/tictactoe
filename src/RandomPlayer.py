import random
from TicTacToe import TicTacToe

def choose_random_move(game: TicTacToe):
    available_moves = game.get_valid_moves()
    action = random.choice(available_moves)
    return action