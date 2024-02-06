import random
from typing import Tuple

from TicTacToe import TicTacToe

class QLearningAgent:

    def __init__(self):
        self.Q = {}
        self.epsilon = 0.7

    def get_q_value(self, state: str, action: Tuple[int,int]) -> float:
        if state not in self.Q:
            return 0.0
        
        if action not in self.Q[state]:
            return 0.0
        
        return self.Q[state][action]
    
    def get_max_q_value(self, state: str):
        if state not in self.Q:
            return 0.0
        
        actions = self.Q[state].values()

        return max(actions)
    
    def set_q_value(self, state, action, q_value):
        if state not in self.Q:
            self.Q[state] = {action: q_value}
            return
        
        self.Q[state][action] = q_value

    def update_epsilon(self,  new_epsilon: float):
        self.epsilon = new_epsilon

    def choose_action(self, game: TicTacToe, training = False):
        available_moves = game.get_valid_moves()
        game_state = game.get_game_state()
        if self.epsilon > 0 and training:
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
    
    def learn_from_game(self, moves, learning_rate: float, reward: int, discount_factor: float):
        (last_state, last_action) = moves[0]
        last_move_q_value = (1 - learning_rate) * self.get_q_value(last_state, last_action) + (learning_rate * reward)

        self.set_q_value(last_state, last_action, last_move_q_value)

        other_moves = moves[1:]
        for index, state in enumerate(other_moves):
            
            if (index == 0):
                next_state = last_state
            else:
                (next_state, _) = other_moves[index - 1]

            (current_state, current_action) = state

            q = (1 - learning_rate) * self.get_q_value(current_state, current_action) + (learning_rate * discount_factor * self.get_max_q_value(next_state))
            self.set_q_value(current_state, current_action, q)


