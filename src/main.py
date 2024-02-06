import random

from QLearningAgent import QLearningAgent
from RandomPlayer import choose_random_move
from ScoreBoard import ScoreBoard
from TicTacToe import CROSS, DRAW, IN_PROGRESS, O_WINS, X_WINS, TicTacToe

# Train a Q learning X player vs O random player
def train(num_games, alpha, epsilon, discount_factor):

    rewards = {
        X_WINS: 1,
        O_WINS: -1,
        DRAW: 0,
        IN_PROGRESS: 0
    }
    agent = QLearningAgent()
    for i in range(num_games):
        if i % 100 == 0:
            print(f'game id: {i}')

        # if (i+1) % (num_games / 10) == 0:
        #     epsilon = max(0, epsilon - 0.1)
        #     agent.update_epsilon(epsilon)
        #     print(f"{i+1}/{num_games} games, using epsilon={epsilon}...")

        game = TicTacToe()
        result = play_training_game(game, agent.choose_action, choose_random_move)

        reward = rewards[result]

        agent.learn_from_game(game.move_history, alpha, reward, discount_factor)


    return agent    

def play_training_game(game: TicTacToe, x_player, o_player):
    while game.in_progress():
        if game.current_player == CROSS:
            action = x_player(game, True)
        else:
            action = o_player(game)
        
        game.make_move(*action)

    return game.game_status


def play(num_games: int, x_player, o_player):
    scoreCard = ScoreBoard()

    for _ in range(num_games):
        result = play_game(x_player, o_player)
        scoreCard.record_result(result)
        
    scoreCard.print_results()


def play_game(x_player, o_player):
    game = TicTacToe()
    return game.play_game(x_player, o_player)

def write_q_table_to_file(model):
    file = open('q_values.txt', 'w')

    for key in model.Q.keys():
        state = key.replace(".", "").replace("1", "X").replace("2", "O").replace("0", ".")
        row1 = state[0:5]
        row2 = state[6:11]
        row3 = state[12:17]
        file.write('----------\n')
        file.write(f'{row1}\n' )
        file.write(f'{row2}\n' )
        file.write(f'{row3}\n' )
        for actions in model.Q[key]:
            file.write(f'{actions} {model.Q[key][actions]}\n')
    file.close()


if __name__ == "__main__":
    EPSILON = 0.7
    DISCOUNT = 1.0
    ALPHA = 0.4
    model = train(1000, ALPHA, EPSILON, DISCOUNT)
    write_q_table_to_file(model)

    print('trained model game')
    play(100, model.choose_action, choose_random_move)

    print('random game')
    play(100, choose_random_move, choose_random_move)


