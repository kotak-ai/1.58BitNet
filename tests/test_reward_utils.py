import unittest
from reward_utils import (
    qa_reward,
    _WN_AVAILABLE,
    reasoning_token_f1,
    step_correctness,
)


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

    @unittest.skipUnless(_WN_AVAILABLE, "WordNet not available")
    def test_synonym(self):
        self.assertGreater(qa_reward("car", "automobile"), 0.0)

    def test_stemming(self):
        self.assertAlmostEqual(qa_reward("running", "run"), 1.0)

    def test_reasoning_token_f1(self):
        pred = "<think>step one</think>42"
        ref_reason = "step one"
        self.assertAlmostEqual(reasoning_token_f1(pred, ref_reason), 1.0)

    def test_step_correctness(self):
        pred = "<think>step one. step two</think>42"
        ref = "step one. step two"
        self.assertAlmostEqual(step_correctness(pred, ref), 1.0)

if __name__ == "__main__":
    unittest.main()
