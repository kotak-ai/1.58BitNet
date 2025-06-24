import unittest
from unittest import mock
import importlib

TRANS_AVAILABLE = importlib.util.find_spec("transformers") is not None

if TRANS_AVAILABLE:
    import inference
    import data_loading_compatibility as dlc
    import evaluation
    import intrinsic_baseline


@unittest.skipUnless(TRANS_AVAILABLE, "transformers not available")
class InferenceCLITest(unittest.TestCase):
    def test_main_calls_run(self):
        with mock.patch('inference.run') as run:
            inference.main(['--model_path', 'm', '--prompt', 'hi'])
            run.assert_called_with('m', ['hi'], max_length=100, evaluate=False)


@unittest.skipUnless(TRANS_AVAILABLE, "transformers not available")
class DataLoadingCLITest(unittest.TestCase):
    def test_main_calls_run(self):
        with mock.patch('data_loading_compatibility.run') as run:
            dlc.main(['--model_path', 'm', '--text', 'hi'])
            run.assert_called_with('m', 'hi', max_length=100)


@unittest.skipUnless(TRANS_AVAILABLE, "transformers not available")
class EvaluationCLITest(unittest.TestCase):
    def test_main_calls_run(self):
        with mock.patch('evaluation.run') as run:
            evaluation.main([
                '--dataset', 'd.json',
                '--ce_model', 'ce',
                '--grpo_model', 'grpo',
            ])
            run.assert_called_with(
                'd.json',
                'ce',
                'grpo',
                max_length=20,
                task='qa',
                two_layer=False,
                guiding_prompt='Review and correct the answer:',
                second_max_length=20,
            )


@unittest.skipUnless(TRANS_AVAILABLE, "transformers not available")
class IntrinsicBaselineCLITest(unittest.TestCase):
    def test_main_calls_run(self):
        with mock.patch('intrinsic_baseline.run') as run:
            intrinsic_baseline.main([
                '--dataset', 'd.json',
                '--model', 'm',
            ])
            run.assert_called_with(
                'd.json',
                'm',
                task='qa',
                max_length=20,
                guiding_prompt='Review and correct the answer:',
                second_max_length=20,
            )


if __name__ == '__main__':
    unittest.main()
