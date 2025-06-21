import unittest
from grpo_data import build_layer1_prompt, build_layer2_prompt

class PromptTemplateTest(unittest.TestCase):
    def test_layer1(self):
        self.assertEqual(
            build_layer1_prompt("Q", system_prompt="SYS"),
            "<system>SYS</system><user>Q</user><assistant>",
        )

    def test_layer2(self):
        text = build_layer2_prompt("Q", "A", "G", system_prompt=None)
        self.assertEqual(text, "<user>Q<think>A</think>G</user><assistant>")

if __name__ == "__main__":
    unittest.main()
