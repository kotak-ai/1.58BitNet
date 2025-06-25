#!/usr/bin/env python3
"""Example showing how the energy RL agent can schedule inference.

The agent interacts with ``EnergyEnv`` and controls when tokens are generated.
A compute action triggers a one token decode step while other actions skip
generation to simulate idling or cooling. This demonstrates how reinforcement
learning could manage energy and thermal constraints during model inference.
"""

import argparse
from typing import Iterable

import torch

from energy_rl import EnergyEnv, QLearningAgent, train
from inference import load_quantized_model


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with scheduling from the RL agent"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=20,
                        help="Maximum tokens to generate")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Training episodes for the agent")
    return parser


def _generate_step(model, tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=1)
    return output


def run(args: argparse.Namespace) -> str:
    env = EnergyEnv()
    agent = QLearningAgent(env)
    train(agent, episodes=args.episodes, max_steps=args.tokens)

    model, tokenizer = load_quantized_model(args.model_path)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(model.lm_head.weight.device)

    state = env.reset()
    generated = input_ids
    for _ in range(args.tokens):
        action = agent.choose_action(state)
        state, _, done = env.step(action)
        if done:
            break
        if action == 1:  # compute step
            generated = _generate_step(model, tokenizer, generated)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    text = run(args)
    print(text)


if __name__ == "__main__":
    main()
