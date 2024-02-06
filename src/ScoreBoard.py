from TicTacToe import DRAW, O_WINS, X_WINS


class ScoreBoard:
    def __init__(self):
        self.results = {
            X_WINS: 0,
            O_WINS: 0,
            DRAW: 0
        }

    def record_result(self, result: int):
        self.results[result] += 1

    def print_results(self):
        print(f'x wins: {self.results[X_WINS]}')
        print(f'o wins: {self.results[O_WINS]}')
        print(f'draws: {self.results[DRAW]}')