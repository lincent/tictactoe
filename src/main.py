import random

from QLearningAgent import QLearningAgent
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
            # X's move (Q learning)
            state = game.get_game_state()
            available_moves = game.get_valid_moves()
            action = agent.choose_action(state, available_moves)

            game.make_move(*action)

            next_state = game.get_game_state()
            # give reward for winning or continuing game
            reward = 1 if game.result == X_WINS or game.result == DRAW else 0

            agent.update_q_value(state, action, reward, next_state, game)

            # Don't try to make O's move if the game is finished
            if game.game_complete:
                break

            # O's move (random)
            available_moves = game.get_valid_moves()
            action = random.choice(available_moves)
            game.make_move(*action)

        # print(game)
        game = TicTacToe()
    return agent

def play(num_games: int, model):
    print('trained game')
    scoreCard = ScoreBoard()

    for i in range(num_games):
        game = TicTacToe()
        while not game.game_complete:
            state = game.get_game_state()
            available_moves = game.get_valid_moves()
            action = model.choose_action(state, available_moves)

            game.make_move(*action)

            if not game.game_complete:
                available_moves = game.get_valid_moves()
                action = random.choice(available_moves)
                game.make_move(*action)
        scoreCard.add_result(game.result)
    scoreCard.print_result()

def random_play(num_games: int):
    print('random game')
    scoreCard = ScoreBoard()

    for i in range(num_games):
        game = TicTacToe()
        while not game.game_complete:
            available_moves = game.get_valid_moves()
            action = random.choice(available_moves)
            game.make_move(*action)

            if not game.game_complete:
                available_moves = game.get_valid_moves()
                action = random.choice(available_moves)
                game.make_move(*action)
        scoreCard.add_result(game.result)
    scoreCard.print_result()


if __name__ == "__main__":
    model = train(30000, 0.5, 0.1, 1)
    play(1000, model)
    random_play(1000)
    # game = TicTacToe("1 0 0 1 -1 -1 0 0 0")
    # action = model.choose_action("1. 0. 0. 1. -1. -1. 0. 0. 0.", game.get_valid_moves())
    # game.make_move(*action)
    # print(game)
    # print(action)

