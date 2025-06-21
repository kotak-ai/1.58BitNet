#!/usr/bin/env python3
"""Train the energy RL agent with configurable environment parameters."""
import argparse
from energy_rl import EnergyEnv, QLearningAgent, train, evaluate, EnvConfig


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the energy RL agent")
    parser.add_argument("--max_energy", type=int, default=10)
    parser.add_argument("--thermal_limit", type=int, default=10)
    parser.add_argument("--harvest_rate", type=float, default=2.0)
    parser.add_argument("--idle_cost", type=float, default=0.5)
    parser.add_argument("--compute_cost", type=float, default=1.5)
    parser.add_argument("--compute_heat", type=float, default=2.0)
    parser.add_argument("--heat_dissipation", type=float, default=1.0)
    parser.add_argument("--cool_cost", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=50,
                        help="Training episodes")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Steps per episode")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    return parser


def run(args: argparse.Namespace) -> tuple[float, float]:
    cfg = EnvConfig(
        max_energy=args.max_energy,
        thermal_limit=args.thermal_limit,
        harvest_rate=args.harvest_rate,
        idle_cost=args.idle_cost,
        compute_cost=args.compute_cost,
        compute_heat=args.compute_heat,
        heat_dissipation=args.heat_dissipation,
        cool_cost=args.cool_cost,
    )
    env = EnergyEnv(cfg)
    agent = QLearningAgent(env, alpha=args.alpha,
                           gamma=args.gamma, epsilon=args.epsilon)
    print("Evaluating untrained agent...")
    pre = evaluate(agent, episodes=5, max_steps=args.max_steps)
    print(f"Average reward before training: {pre:.2f}")
    train(agent, episodes=args.episodes, max_steps=args.max_steps)
    post = evaluate(agent, episodes=5, max_steps=args.max_steps)
    print(f"Average reward after training: {post:.2f}")
    return pre, post


def main(argv: list[str] | None = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
