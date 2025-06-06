import unittest
from energy_rl import EnergyEnv, QLearningAgent, train


class EnergyRLTest(unittest.TestCase):
    def test_q_learning_update(self):
        env = EnergyEnv()
        agent = QLearningAgent(env, epsilon=0.5)
        original = agent.q_table.copy()
        train(agent, episodes=2, max_steps=5)
        self.assertTrue((agent.q_table != original).any())
        self.assertEqual(agent.q_table.shape[0], env.state_space)
        self.assertEqual(agent.q_table.shape[1], env.action_space)


if __name__ == "__main__":
    unittest.main()
