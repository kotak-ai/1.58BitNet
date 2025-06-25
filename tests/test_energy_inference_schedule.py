import unittest
from unittest import mock
import importlib

TRANS_AVAILABLE = importlib.util.find_spec("transformers") is not None

if TRANS_AVAILABLE:
    import energy_inference_schedule as eis


@unittest.skipUnless(TRANS_AVAILABLE, "transformers not available")
class EnergyInferenceScheduleCLITest(unittest.TestCase):
    def test_main_calls_run(self):
        with mock.patch('energy_inference_schedule.run') as run:
            eis.main(['--model_path', 'm', '--prompt', 'hi'])
            run.assert_called()


if __name__ == '__main__':
    unittest.main()
