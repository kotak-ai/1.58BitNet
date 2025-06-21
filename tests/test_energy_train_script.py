import unittest
from energy_rl_train import get_arg_parser, run


class EnergyTrainScriptTest(unittest.TestCase):
    def test_reward_improves(self):
        parser = get_arg_parser()
        args = parser.parse_args([
            '--episodes', '30',
            '--max_steps', '10',
        ])
        pre, post = run(args)
        self.assertGreaterEqual(post, pre)


if __name__ == '__main__':
    unittest.main()
