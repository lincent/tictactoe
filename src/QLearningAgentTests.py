import unittest

from QLearningAgent import QLearningAgent

class QLearningAgentTests(unittest.TestCase):

    def test_add_q_value(self):
        sut = QLearningAgent()
        sut.set_q_value("state", (1,2), 1.3)

        self.assertEqual(sut.Q["state"][(1,2)], 1.3)

    def test_add_second_action_to_state(self):
        sut = QLearningAgent()
        sut.set_q_value("state", (1,2), 1.3)
        sut.set_q_value("state", (0,2), 2.3)

        self.assertEqual(sut.Q["state"][(0,2)], 2.3)

    def test_get_q_value_no_state(self):
        sut = QLearningAgent()
        result = sut.get_q_value("state", (1,2))

        self.assertEqual(result, 0.0)

    def test_get_max_q(self):
        sut = QLearningAgent()
        sut.set_q_value("any", (0,0), 0.1)
        sut.set_q_value("any", (0,1), 1)
        sut.set_q_value("any", (0,2), 0.3)
        sut.set_q_value("any", (0,3), 0.6)

        result = sut.get_max_q_value("any")

        self.assertEqual(1, result)

    def test_get_q_value_no_action(self):
        sut = QLearningAgent()
        sut.set_q_value("state", (2,2), 1.2)
        result = sut.get_q_value("state", (1,2))

        self.assertEqual(result, 0.0)

    def test_get_q_value(self):
        sut = QLearningAgent()
        sut.set_q_value("state", (2,2), 1.2)
    
        result = sut.get_q_value("state", (2,2))

        self.assertEqual(result, 1.2)

    def test_learn(self):
        moves = [("1 1 0", (0,2)), ("1 0 0", (0,1)), ("0 0 0", (0,0))]
        alpha = 0.9
        reward = 1
        discount_factor = 1

        sut = QLearningAgent()
        sut.learn_from_game(moves, alpha, reward, discount_factor)

        start = sut.get_q_value("0 0 0", (0,0))
        self.assertAlmostEqual(0.729, start)

        mid = sut.get_q_value("1 0 0", (0,1))
        self.assertEqual(0.81, mid)

        end = sut.get_q_value("1 1 0", (0,2))
        self.assertEqual(0.9, end)

    def test_learn2(self):
        moves = [("1 1 0", (0,2)), ("1 0 0", (0,1)), ("0 0 0", (0,0))]
        moves2 = [("1 1 0", (1,2)), ("1 0 0", (0,1)), ("0 0 0", (0,0))]
        alpha = 0.9
        discount_factor = 1

        sut = QLearningAgent()
        sut.learn_from_game(moves, alpha, 1, discount_factor)

        start = sut.get_q_value("0 0 0", (0,0))
        self.assertAlmostEqual(0.729, start)

        mid = sut.get_q_value("1 0 0", (0,1))
        self.assertEqual(0.81, mid)

        end = sut.get_q_value("1 1 0", (0,2))
        self.assertEqual(0.9, end)

        sut.learn_from_game(moves2, alpha, -1, discount_factor)



if __name__ == '__main__':
    unittest.main()