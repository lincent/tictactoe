import random

from QLearningAgent import QLearningAgent
from RandomPlayer import choose_random_move
from ScoreBoard import ScoreBoard
from TicTacToe import DRAW, X_WINS, TicTacToe

# Train a Q learning X player vs O random player
def train(num_episodes, alpha, epsilon, discount_factor):
    agent = QLearningAgent(alpha, epsilon, discount_factor)
    
    game = TicTacToe()
    for i in range(num_episodes):
        if i % 5000 == 0:
            print(f'game id: {i}')
        while not game.game_complete:
            action = agent.choose_action(game)
            game.make_move(*action)

            # give reward for winning or continuing game
            next_state = game.get_game_state()
            reward = 1 if game.result == X_WINS or game.result == DRAW else 0

            agent.update_q_value(game.get_game_state, action, reward, next_state, game)

            # Don't try to make O's move if the game is finished
            if game.game_complete:
                break

            # O's move (random)
            action = choose_random_move(game)
            game.make_move(*action)

        # print(game)
        game = TicTacToe()
    return agent

def play(num_games: int, model: QLearningAgent):
    print('trained game')
    scoreCard = ScoreBoard()

    for i in range(num_games):
        game = TicTacToe()
        while not game.game_complete:
            action = model.choose_action(game)
            game.make_move(*action)

            if not game.game_complete:
                action = choose_random_move(game)
                game.make_move(*action)
        scoreCard.record_result(game.result)
    scoreCard.print_results()

def random_play(num_games: int):
    print('random game')
    scoreCard = ScoreBoard()

    for i in range(num_games):
        game = TicTacToe()
        while not game.game_complete:
            action = choose_random_move(game)
            game.make_move(*action)

            if not game.game_complete:
                action = choose_random_move(game)
                game.make_move(*action)
        scoreCard.record_result(game.result)
    scoreCard.print_results()


if __name__ == "__main__":
    model = train(10000, 0.5, 0.1, 1)
    play(1000, model)
    random_play(1000)
    # game = TicTacToe("1 0 0 1 -1 -1 0 0 0")
    # action = model.choose_action("1. 0. 0. 1. -1. -1. 0. 0. 0.", game.get_valid_moves())
    # game.make_move(*action)
    # print(game)
    # print(action)

