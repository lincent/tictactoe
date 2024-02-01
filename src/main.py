import random

from QLearningAgent import QLearningAgent
from TicTacToe import TicTacToe, CROSS

class ScoreStore:
    x_wins: int
    o_wins: int
    draws: int

    def __init__(self):
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0

    def add_result(self, game: TicTacToe):
        if game.winner == -1:
            self.o_wins = self.o_wins + 1
        if game.winner == 1:
            self.x_wins = self.x_wins + 1
        if game.winner == 0:
            self.draws = self.draws + 1

    def print_result(self):
        print(f'x wins: {self.x_wins}')
        print(f'o wins: {self.o_wins}')
        print(f'draws: {self.draws}')


# Train a Q learning X player vs O random player
def train(num_episodes, alpha, epsilon, discount_factor):
    agent = QLearningAgent(alpha, epsilon, discount_factor)
    
    game = TicTacToe()
    for i in range(num_episodes):
        if i % 5000 == 0:
            print(f'game id: {i}')
        while not game.over:
            # X's move (Q learning)
            state = game.hash()
            available_moves = game.valid_moves()
            action = agent.choose_action(state, available_moves)

            game.make_move(*action)

            next_state = game.hash()
            # give reward for winning or continuing game
            reward = 1 if game.winner == CROSS or game.winner == 0 else 0

            agent.update_q_value(state, action, reward, next_state, game)

            # Don't try to make O's move if the game is finished
            if game.over:
                break

            # O's move (random)
            available_moves = game.valid_moves()
            action = random.choice(available_moves)
            game.make_move(*action)

        # print(game)
        game = TicTacToe()
    return agent

def play(num_games: int, model):
    print('trained game')
    scoreCard = ScoreStore()

    for i in range(num_games):
        game = TicTacToe()
        while not game.over:
            state = game.hash()
            available_moves = game.valid_moves()
            action = model.choose_action(state, available_moves)

            game.make_move(*action)

            if not game.over:
                available_moves = game.valid_moves()
                action = random.choice(available_moves)
                game.make_move(*action)
        scoreCard.add_result(game)
    scoreCard.print_result()

def random_play(num_games: int):
    print('random game')
    scoreCard = ScoreStore()

    for i in range(num_games):
        game = TicTacToe()
        while not game.over:
            available_moves = game.valid_moves()
            action = random.choice(available_moves)
            game.make_move(*action)

            if not game.over:
                available_moves = game.valid_moves()
                action = random.choice(available_moves)
                game.make_move(*action)
        scoreCard.add_result(game)
    scoreCard.print_result()


if __name__ == "__main__":
    model = train(30000, 0.5, 0.1, 1)
    play(1000, model)
    random_play(1000)
    game = TicTacToe("[1 0 0 1 -1 -1 0 0 0]")
    action = model.choose_action("[1. 0. 0. 1. -1. -1. 0. 0. 0.]", game.valid_moves())
    game.make_move(*action)
    print(game)
    print(action)

