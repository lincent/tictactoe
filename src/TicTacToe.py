import numpy as np

## square values
CROSS = 1
NOUGHT = -1
EMPTY_SQUARE = 0

## game status states
IN_PROGRESS = 2
X_WINS = 1
O_WINS = -1
DRAW = 0

BOARD_SIZE = 3


class TicTacToe:

    current_player = CROSS
    game_status = IN_PROGRESS
    move_history = []

    def __init__(self, board=None):
        if board is not None:
            self.board = self.convert_game_state_to_board(board)
        else:
            self.board = np.zeros([BOARD_SIZE, BOARD_SIZE])

    def play_game(self, x_player, o_player):
        while self.in_progress():
            if self.current_player == CROSS:
                action = x_player(self)
            else:
                action = o_player(self)
            
            self.make_move(*action)

        return self.game_status

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

    def get_game_status(self):
        return self.game_status
    
    def is_game_complete(self):
        return self.game_status != IN_PROGRESS
    
    def in_progress(self):
        return self.game_status == IN_PROGRESS

    def get_valid_moves(self):
        if self.game_status != IN_PROGRESS:
            return []
        return [tuple(move) for move in np.argwhere(self.board == EMPTY_SQUARE).tolist()]

    def make_move(self, row, col):
        if self.board[row, col] != EMPTY_SQUARE:
            return
        
        self.board[row, col] = self.current_player

        self.store_move()
        self.update_game_status()
        self.change_player()

        return self.game_status

    def change_player(self):
        self.current_player = NOUGHT if self.current_player == CROSS else CROSS

    def store_move(self):
        self.move_history.append(self.get_game_state())

    def update_game_status(self):
        # rows and columns
        for i in range(BOARD_SIZE):
            row = self.board[i]
            col = self.board[:, i]

            if np.all(row == CROSS) or np.all(col == CROSS):
                self.game_status = X_WINS
                return

            if np.all(row == NOUGHT) or np.all(col == NOUGHT):
                self.game_status = O_WINS
                return

        # diagonals
        diagonal_1 = self.board.diagonal()
        diagonal_2 = np.fliplr(self.board).diagonal()
        if np.all(diagonal_1 == CROSS) or np.all(diagonal_2 == CROSS):
            self.game_status = X_WINS
            return
        if np.all(diagonal_1 == NOUGHT) or np.all(diagonal_2 == NOUGHT):
            self.game_status = O_WINS
            return
        
        if len(self.get_valid_moves()) == 0:
            self.game_status = DRAW
            return
        
        self.game_status = IN_PROGRESS
        return
