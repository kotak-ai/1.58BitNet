import unittest
from grpo_data import build_layer1_prompt, build_layer2_prompt

class PromptTemplateTest(unittest.TestCase):
    def test_layer1(self):
        expected = (
            "<|im_start|>system\nSYS\n<|im_end|>\n"
            "<|im_start|>user\nQ\n<|im_end|>\n<|im_start|>assistant"
        )
        self.assertEqual(build_layer1_prompt("Q", system_prompt="SYS"), expected)

    def test_layer2(self):
        text = build_layer2_prompt("Q", "A", "G", system_prompt=None)
        expected = (
            "<|im_start|>system\nYou are a helpful AI assistant tasked with reviewing and correcting solutions.\n"
            "The User will provide a problem and an attempted solution. Your job is to identify any errors "
            "and provide a corrected solution if needed. Always show your reasoning process.\n<|im_end|>\n"
            "<|im_start|>user\nQ\n<|im_end|>\n<|im_start|>assistant\nA\nG\n<|im_end|>\n<|im_start|>assistant"
        )
        self.assertEqual(text, expected)

if __name__ == "__main__":
    unittest.main()
