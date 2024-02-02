import numpy as np

CROSS = 1
NOUGHT = -1
EMPTY = 0

X_WINS = 1
O_WINS = -1
DRAW = 0

BOARD_SIZE = 3


class TicTacToe:

    game_complete = False
    current_player = CROSS
    result = EMPTY
    move_history = []

    def __init__(self, board=None):
        if board is not None:
            self.board = self.convert_game_state_to_board(board)
        else:
            self.board = np.zeros([BOARD_SIZE, BOARD_SIZE])

    def get_game_state(self):
        return str(self.board.reshape(9)).rstrip(']').lstrip('[')
    
    def convert_game_state_to_board(self, state):
        if type(state) == str:
            return np.array(list(map(int, state.split(' ')))).reshape([3,3])
        return state

    def print_board(self):
        s = ""
        for row in self.board:
            for i in row:
                s += 'X' if i == CROSS else 'O' if i == NOUGHT else '.'
            s += "\n"

        print(s)

    def get_valid_moves(self):
        if self.result != 0:
            return []
        return [tuple(move) for move in np.argwhere(self.board == 0.).tolist()]

    def make_move(self, row, col):
        if self.board[row, col] != 0.:
            return
        
        self.board[row, col] = self.current_player

        self.store_move()

        result = self.check_if_game_over()
        if (result != None):
            self.result = result
            return result

        self.change_player()

    def change_player(self):
        self.current_player = NOUGHT if self.current_player == CROSS else CROSS

    def store_move(self):
        self.move_history.append(self.get_game_state())

    def check_if_game_over(self) -> int | None: 
        # rows and columns
        for i in range(BOARD_SIZE):
            row = self.board[i]
            col = self.board[:, i]

            if np.all(row == CROSS) or np.all(col == CROSS):
                self.mark_game_complete()
                return X_WINS

            if np.all(row == NOUGHT) or np.all(col == NOUGHT):
                self.mark_game_complete()
                return O_WINS

        # diagonals
        diagonal_1 = self.board.diagonal()
        diagonal_2 = np.fliplr(self.board).diagonal()
        if np.all(diagonal_1 == CROSS) or np.all(diagonal_2 == CROSS):
            self.mark_game_complete()
            return X_WINS
        if np.all(diagonal_1 == NOUGHT) or np.all(diagonal_2 == NOUGHT):
            self.mark_game_complete()
            return O_WINS
        
        if len(self.get_valid_moves()) == 0:
            self.mark_game_complete()
            return DRAW
        
        return
        
    def mark_game_complete(self): 
        self.game_complete = True


if __name__ == '__main__':
    game = TicTacToe()

    while not game.game_complete:
        print(game)
        move = input("Player turn. Enter row and column: ")
        x, y = map(int, move.split())
        game.make_move(x, y)

    print(game)
    if game.result != 0:
        winner = 'X' if game.result == X_WINS else 'O'
        print(f"{winner} is the winner")
    else:
        print('Game is a draw')
