import random

from TicTacToe import TicTacToe

class QLearningAgent:

    def __init__(self, alpha: float, epsilon: float, discount_factor: float):
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    def get_q_value(self, state: str, action):
        key = (state, action)
        if key not in self.Q:
            self.Q[key] = 0.0
        return self.Q[key]

    def choose_action(self, game: TicTacToe):
        available_moves = game.get_valid_moves()
        game_state = game.get_game_state()
        if self.epsilon > 0:
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(available_moves)

        q_values = [self.get_q_value(game_state, action) for action in available_moves]
        max_Q = max(q_values)
        if q_values.count(max_Q) > 1:
            best_moves = [i for i in range(len(available_moves)) if q_values[i] == max_Q]
            i = random.choice(best_moves)
        else:
            i = q_values.index(max_Q)
        return available_moves[i]

    def update_q_value(self, state: str, action, reward, next_state: str, game: TicTacToe):
        valid_moves = game.get_valid_moves()
        next_q_values = [self.get_q_value(next_state, next_action) for next_action in valid_moves]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        self.get_q_value(state, action)
        self.Q[(state, action)] += self.alpha * (reward + self.discount_factor * max_next_q - self.Q[(state, action)])
