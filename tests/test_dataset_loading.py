import unittest
from unittest import mock

import grpo_data
import trainingv2

class DatasetLoadingTest(unittest.TestCase):
    def test_success(self):
        fake_ds = [{'problem': 'q', 'solution': 'a'}]
        with mock.patch('datasets.load_dataset', return_value=fake_ds) as ld:
            data = grpo_data.load_math_dataset(split='x')
            ld.assert_called_with('hendrycks/math', split='x')
            self.assertEqual(data, [{'query': 'q', 'answer': 'a'}])

    def test_local_path(self):
        fake_ds = [{'question': 'q', 'answer': 'a'}]
        with mock.patch('datasets.load_dataset', return_value=fake_ds) as ld:
            data = grpo_data.load_gsm8k_dataset(path='local')
            ld.assert_called_with('local', split='test')
            self.assertEqual(data, [{'query': 'q', 'answer': 'a'}])

    def test_failure(self):
        with mock.patch('datasets.load_dataset', side_effect=OSError):
            with self.assertRaises(RuntimeError):
                grpo_data.load_minerva_math_dataset()

    def test_trainingv2_failure(self):
        with mock.patch('datasets.load_dataset', side_effect=OSError):
            with self.assertRaises(RuntimeError):
                trainingv2.download_dataset('bad')

if __name__ == '__main__':
    unittest.main()
