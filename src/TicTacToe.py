import numpy as np

CROSS = 1
NOUGHT = -1

BOARD_SIZE = 3


class TicTacToe:

    def __init__(self, board=None):
        if board is not None:
            self.board = self.dehash(board)
        else:
            self.board = np.zeros([BOARD_SIZE, BOARD_SIZE])
        self.player = CROSS
        self.over = False
        self.winner = 0
        self.move_history = []

    def hash(self):
        
        return str(self.board.reshape(9))
    
    def dehash(self, hash):
        if type(hash) == str:
            hash = hash.rstrip(']')
            hash = hash.lstrip('[')
            return np.array(list(map(int, hash.split(' ')))).reshape([3,3])
        return hash

    def __repr__(self):
        s = ""
        for row in self.board:
            for i in row:
                s += 'X' if i == CROSS else 'O' if i == NOUGHT else '.'
            s += "\n"

        return s

    def valid_moves(self):
        if self.winner != 0:
            return []
        return [tuple(move) for move in np.argwhere(self.board == 0.).tolist()]

    def make_move(self, row, col):
        if self.board[row, col] != 0.:
            return
        self.board[row, col] = self.player

        self.move_history.append(self.hash())

        winner = self.check_winner()
        if winner != 0:
            self.over = True
            self.winner = winner
            return

        if len(self.valid_moves()) == 0:
            self.over = True
            return

        self.player = NOUGHT if self.player == CROSS else CROSS


    def check_winner(self):
        # rows and columns
        for i in range(BOARD_SIZE):
            row = self.board[i]
            col = self.board[:, i]

            if np.all(row == CROSS) or np.all(col == CROSS):
                return CROSS

            if np.all(row == NOUGHT) or np.all(col == NOUGHT):
                return NOUGHT

        # diagonals
        diagonal_1 = self.board.diagonal()
        diagonal_2 = np.fliplr(self.board).diagonal()
        if np.all(diagonal_1 == CROSS) or np.all(diagonal_2 == CROSS):
            return CROSS
        if np.all(diagonal_1 == NOUGHT) or np.all(diagonal_2 == NOUGHT):
            return NOUGHT

        return 0


if __name__ == '__main__':
    game = TicTacToe()

    while not game.over:
        print(game)
        move = input("Player turn. Enter row and column: ")
        x, y = map(int, move.split())
        game.make_move(x, y)

    print(game)
    if game.winner != 0:
        winner = 'X' if game.winner == CROSS else 'O'
        print(f"{winner} is the winner")
    else:
        print('Game is a draw')
