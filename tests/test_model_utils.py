import unittest
import torch.nn as nn
from model_utils import count_parameters, verify_parameter_count


class ModelUtilsTest(unittest.TestCase):
    def test_count_parameters(self):
        model = nn.Linear(2, 3)
        self.assertEqual(count_parameters(model), 2 * 3 + 3)

    def test_verify_passes_within_tolerance(self):
        model = nn.Linear(2, 2)
        actual = count_parameters(model)
        # Should not raise for exact value
        verify_parameter_count(actual, actual)
        # Should pass with difference below tolerance
        verify_parameter_count(actual + 1, actual, tolerance_ratio=0.2)

    def test_verify_raises_on_large_difference(self):
        with self.assertRaises(ValueError):
            verify_parameter_count(120, 100, tolerance_ratio=0.01)


if __name__ == '__main__':
    unittest.main()
