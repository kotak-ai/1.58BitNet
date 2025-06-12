import argparse
import json
import unittest
from grpo_train import get_arg_parser, update_args_with_config

class ConfigTest(unittest.TestCase):
    def test_apply_config(self):
        parser = get_arg_parser()
        cfg = {"lr": 0.5, "group_size": 4, "guiding_prompt": "check"}
        with open("tmp_cfg.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        args = parser.parse_args(["--dataset", "d.json", "--model_path", "m", "--config", "tmp_cfg.json"])
        update_args_with_config(args, parser)
        self.assertEqual(args.lr, 0.5)
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.guiding_prompt, "check")
        args = parser.parse_args(["--dataset", "d.json", "--model_path", "m", "--config", "tmp_cfg.json", "--lr", "0.1"])
        update_args_with_config(args, parser)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.guiding_prompt, "check")

if __name__ == "__main__":
    unittest.main()
