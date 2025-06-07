import unittest
from reward_utils import qa_reward


class RewardUtilsTest(unittest.TestCase):
    def test_exact_match(self):
        pred = "Paris is the capital of France."
        ref = "Paris is the capital of France"
        self.assertAlmostEqual(qa_reward(pred, ref), 1.0)

    def test_partial_match(self):
        pred = "Paris"
        ref = "Paris is the capital of France"
        val = qa_reward(pred, ref)
        self.assertGreater(val, 0.0)
        self.assertLess(val, 1.0)

    def test_synonym(self):
        self.assertGreater(qa_reward("car", "automobile"), 0.0)

    def test_stemming(self):
        self.assertAlmostEqual(qa_reward("running", "run"), 1.0)


if __name__ == "__main__":
    unittest.main()
