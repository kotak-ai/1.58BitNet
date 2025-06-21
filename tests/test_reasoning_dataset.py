import unittest
from grpo_data import load_qa_dataset

class ReasoningDatasetTest(unittest.TestCase):
    def test_load_reasoning(self):
        data = load_qa_dataset('tests/data/reasoning_dataset.json')
        self.assertIn('reasoning', data[0])
        self.assertEqual(data[0]['reasoning'], 'step one. step two.')

if __name__ == '__main__':
    unittest.main()
