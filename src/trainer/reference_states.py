"""Adaptive reference-state initialization from valid expert trajectories."""

from __future__ import annotations

import numpy as np

from src.config.demo import EnvironmentConfig
from src.envs.grid_robot import GridRobotEnv, Observation, ReferenceState


class ReferenceStateSampler:
    """Revisit demonstration phases where current rollouts often fail."""

    def __init__(
        self,
        environment_config: EnvironmentConfig,
        seed: int,
        reference_probability: float = 0.6,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.reference_probability = reference_probability
        env = GridRobotEnv(environment_config)
        states: list[ReferenceState] = []
        for start in env.starts:
            env.reset(reference=ReferenceState(start, False, 0))
            while not env.done:
                # Exclude terminal states and keep the original start in the mix.
                states.append(env.reference_state())
                env.step(env.expert_action())
        self.states = tuple(dict.fromkeys(states))
        self.visits = np.zeros(len(self.states), dtype=np.float64)
        self.failures = np.zeros(len(self.states), dtype=np.float64)

    def reset_episode(
        self, env: GridRobotEnv, seed: int | None = None
    ) -> tuple[Observation, int | None]:
        """Mix full episodes with reference starts sampled by estimated difficulty."""
        if self.rng.random() >= self.reference_probability:
            return env.reset(seed=seed), None
        # A uniform floor prevents the sampler from forgetting easy phases.
        failure_rate = (self.failures + 1.0) / (self.visits + 2.0)
        probabilities = failure_rate + 0.15
        probabilities /= probabilities.sum()
        index = int(self.rng.choice(len(self.states), p=probabilities))
        return env.reset(seed=seed, reference=self.states[index]), index

    def observe_outcome(self, index: int | None, success: bool) -> None:
        """Update phase difficulty only after an episode actually terminates."""
        if index is None:
            return
        self.visits[index] += 1.0
        self.failures[index] += float(not success)
