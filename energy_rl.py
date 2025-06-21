import numpy as np
from dataclasses import dataclass
from energy_utils import compute_energy_metrics, energy_utility


@dataclass
class EnvConfig:
    max_energy: int = 10
    thermal_limit: int = 10
    harvest_rate: float = 2.0
    idle_cost: float = 0.5
    compute_cost: float = 1.5
    compute_heat: float = 2.0
    heat_dissipation: float = 1.0
    cool_cost: float = 0.2


class EnergyEnv:
    """Simple environment with discrete energy and thermal states."""

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.cfg = config
        self.action_space = 4  # idle, compute, harvest, cool
        self.state_space = (self.cfg.max_energy + 1) * (self.cfg.thermal_limit + 1)
        self.reset()

    def _state_index(self):
        e = int(np.clip(round(self.energy), 0, self.cfg.max_energy))
        t = int(np.clip(round(self.temperature), 0, self.cfg.thermal_limit))
        return e * (self.cfg.thermal_limit + 1) + t

    def reset(self):
        self.energy = float(self.cfg.max_energy)
        self.temperature = 0.0
        return self._state_index()

    def step(self, action: int):
        harvested = 0.0
        consumed = 0.0
        external_reward = 0.0
        if action == 0:  # idle
            consumed = self.cfg.idle_cost
            self.temperature = max(0.0, self.temperature - self.cfg.heat_dissipation)
        elif action == 1:  # compute
            consumed = self.cfg.compute_cost
            external_reward = 1.0
            self.temperature += self.cfg.compute_heat
        elif action == 2:  # harvest
            harvested = self.cfg.harvest_rate
        elif action == 3:  # cool
            consumed = self.cfg.cool_cost
            self.temperature = max(0.0, self.temperature - 2 * self.cfg.heat_dissipation)
        else:
            raise ValueError("Invalid action")

        self.energy = np.clip(self.energy + harvested - consumed, 0.0, self.cfg.max_energy)
        # Natural cooling
        self.temperature = max(0.0, self.temperature - 0.1)

        evs, tri, she = compute_energy_metrics(
            self.energy,
            self.cfg.max_energy,
            harvested,
            consumed,
            self.temperature,
            self.cfg.thermal_limit,
        )
        util = energy_utility(evs, tri, she)
        reward = util + external_reward
        done = self.energy <= 0 or self.temperature >= self.cfg.thermal_limit
        return self._state_index(), reward, done


class QLearningAgent:
    """Tabular Q-learning agent for the energy environment."""

    def __init__(self, env: EnergyEnv, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.state_space, env.action_space))

    def choose_action(self, state: int):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state, action]
        )


def train(agent: QLearningAgent, episodes=10, max_steps=50):
    for _ in range(episodes):
        state = agent.env.reset()
        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = agent.env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break
    return agent


def evaluate(agent: QLearningAgent, episodes: int = 5, max_steps: int = 50) -> float:
    """Run episodes using the greedy policy and return the average reward."""
    total = 0.0
    for _ in range(episodes):
        state = agent.env.reset()
        for _ in range(max_steps):
            action = int(np.argmax(agent.q_table[state]))
            state, reward, done = agent.env.step(action)
            total += reward
            if done:
                break
    return total / episodes
