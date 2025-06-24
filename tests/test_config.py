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
        self.assertEqual(args.guiding_prompt, ["check"])
        args = parser.parse_args(["--dataset", "d.json", "--model_path", "m", "--config", "tmp_cfg.json", "--lr", "0.1"])
        update_args_with_config(args, parser)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.guiding_prompt, ["check"])

    def test_guiding_prompt_file(self):
        parser = get_arg_parser()
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write("file prompt")
        cfg = {"guiding_prompt": "prompt.txt"}
        with open("tmp_cfg2.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        args = parser.parse_args(["--dataset", "d.json", "--model_path", "m", "--config", "tmp_cfg2.json"])
        update_args_with_config(args, parser)
        self.assertEqual(args.guiding_prompt, ["file prompt"])

    def test_guiding_prompt_list_file(self):
        parser = get_arg_parser()
        with open("prompts.txt", "w", encoding="utf-8") as f:
            f.write("one\ntwo\n")
        cfg = {"guiding_prompt": "prompts.txt"}
        with open("tmp_cfg3.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        args = parser.parse_args(["--dataset", "d.json", "--model_path", "m", "--config", "tmp_cfg3.json"])
        update_args_with_config(args, parser)
        self.assertEqual(args.guiding_prompt, ["one", "two"])

    def test_guiding_prompt_cli_file(self):
        parser = get_arg_parser()
        with open("cli_prompts.txt", "w", encoding="utf-8") as f:
            f.write("a\nb\n")
        args = parser.parse_args([
            "--dataset",
            "d.json",
            "--model_path",
            "m",
            "--guiding_prompt",
            "cli_prompts.txt",
        ])
        update_args_with_config(args, parser)
        self.assertEqual(args.guiding_prompt, ["a", "b"])

    def test_multiple_reward_models_cli(self):
        parser = get_arg_parser()
        args = parser.parse_args([
            "--dataset",
            "d.json",
            "--model_path",
            "m",
            "--reward_model",
            "a.pt",
            "b.pt",
            "--reward_weights",
            "0.4",
            "0.6",
        ])
        self.assertEqual(args.reward_model, ["a.pt", "b.pt"])
        self.assertEqual(args.reward_weights, [0.4, 0.6])

    def test_improvement_threshold(self):
        parser = get_arg_parser()
        cfg = {"improvement_threshold": 0.2}
        with open("cfg_thr.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        args = parser.parse_args([
            "--dataset",
            "d.json",
            "--model_path",
            "m",
            "--config",
            "cfg_thr.json",
        ])
        update_args_with_config(args, parser)
        self.assertEqual(args.improvement_threshold, 0.2)
        args = parser.parse_args([
            "--dataset",
            "d.json",
            "--model_path",
            "m",
            "--improvement_threshold",
            "0.1",
        ])
        update_args_with_config(args, parser)
        self.assertEqual(args.improvement_threshold, 0.1)
if __name__ == "__main__":
    unittest.main()
