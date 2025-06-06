import unittest
from energy_utils import compute_energy_metrics, energy_utility


class EnergyUtilsTest(unittest.TestCase):
    def test_metrics_range(self):
        evs, tri, she = compute_energy_metrics(
            energy_level=5,
            energy_capacity=10,
            energy_harvested=2,
            energy_consumed=1,
            thermal_load=3,
            thermal_limit=5,
        )
        self.assertGreaterEqual(evs, 0.0)
        self.assertLessEqual(evs, 1.0)
        self.assertGreaterEqual(tri, 0.0)
        self.assertLessEqual(tri, 1.0)
        self.assertGreaterEqual(she, 0.0)

    def test_utility_output(self):
        evs, tri, she = 0.5, 0.5, 5
        util = energy_utility(evs, tri, she)
        self.assertIsInstance(util, float)


if __name__ == "__main__":
    unittest.main()
