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

        game = TicTacToe()
        while not game.is_game_complete():
            action = agent.choose_action(game)
            game.make_move(*action)

            # give reward for winning or continuing game
            next_state = game.get_game_state()
            reward = 1 if game.game_status == X_WINS or game.game_status == DRAW else 0

            agent.update_q_value(game.get_game_state, action, reward, next_state, game)

            # Don't try to make O's move if the game is finished
            if game.is_game_complete():
                break

            # O's move (random)
            action = choose_random_move(game)
            game.make_move(*action)

        # print(game)
    return agent

def play(num_games: int, x_player, o_player):
    scoreCard = ScoreBoard()

    for _ in range(num_games):
        result = play_game(x_player, o_player)
        scoreCard.record_result(result)
        
    scoreCard.print_results()


def play_game(x_player, o_player):
    game = TicTacToe()
    return game.play_game(x_player, o_player)


if __name__ == "__main__":
    # model = train(10000, 0.5, 0.1, 1)
    # print('trained model game')
    # play(1000, model.choose_action, choose_random_move)

    print('random game')
    play(1000, choose_random_move, choose_random_move)
