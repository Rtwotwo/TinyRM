"""A compact grid robot with a key, goal, walls, and hazardous cells."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.config.demo import EnvironmentConfig


REWARD_FEATURE_NAMES = (
    "progress",
    "key_pickup",
    "goal_completion",
    "hazard_contact",
    "wall_collision",
    "time_cost",
)
ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Observation:
    """Public state available to the policy and value function."""

    grid: np.ndarray
    scalars: np.ndarray


@dataclass(frozen=True)
class ReferenceState:
    """A physically valid state saved from an expert demonstration."""

    position: tuple[int, int]
    has_key: bool
    steps: int


@dataclass(frozen=True)
class StepResult:
    """Transition data with an evaluation-only oracle return."""

    observation: Observation
    features: np.ndarray
    context: np.ndarray
    oracle_reward: float
    done: bool
    success: bool


class GridRobotEnv:
    """Navigate around walls and hazards, collect a key, then reach a goal."""

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self.size = config.size
        self.walls = frozenset(
            (row, 3) for row in range(self.size) if row not in (2, 5)
        )
        self.hazards = frozenset({(2, 2), (4, 2), (3, 5)})
        self.key = (5, 1)
        self.goal = (1, 5)
        self.starts = ((1, 1), (1, 2), (2, 1), (4, 1), (5, 0))
        self._rng = np.random.default_rng()
        self.position = self.starts[0]
        self.has_key = False
        self.steps = 0
        self.done = False

    def reset(
        self,
        seed: int | None = None,
        reference: ReferenceState | None = None,
    ) -> Observation:
        """Start at an ordinary position or a valid demonstration state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if reference is None:
            self.position = self.starts[int(self._rng.integers(len(self.starts)))]
            self.has_key = False
            self.steps = 0
        else:
            if (
                not self._walkable(reference.position)
                or reference.steps < 0
                or reference.steps >= self.config.max_steps
            ):
                raise ValueError("Reference state is outside the valid map or horizon.")
            self.position = reference.position
            self.has_key = reference.has_key
            self.steps = reference.steps
        self.done = False
        return self.observation()

    def reference_state(self) -> ReferenceState:
        """Snapshot a reachable state for reference-state initialization."""
        return ReferenceState(self.position, self.has_key, self.steps)

    def observation(self) -> Observation:
        """Encode geometry as image channels and task state as scalars."""
        grid = np.zeros((5, self.size, self.size), dtype=np.float32)
        grid[0, self.position[0], self.position[1]] = 1.0
        for row, col in self.walls:
            grid[1, row, col] = 1.0
        for row, col in self.hazards:
            grid[2, row, col] = 1.0
        if not self.has_key:
            grid[3, self.key[0], self.key[1]] = 1.0
        grid[4, self.goal[0], self.goal[1]] = 1.0
        target = self.goal if self.has_key else self.key
        distance = self._path_distance(self.position, target)
        scalars = np.asarray(
            (
                float(self.has_key),
                (self.config.max_steps - self.steps) / self.config.max_steps,
                distance / (2 * (self.size - 1)),
                self.position[0] / (self.size - 1),
                self.position[1] / (self.size - 1),
            ),
            dtype=np.float32,
        )
        return Observation(grid=grid, scalars=scalars)

    def step(self, action: int) -> StepResult:
        """Advance one cell and expose interpretable transition features."""
        if self.done:
            raise RuntimeError("Reset the environment after an episode ends.")
        if action < 0 or action >= len(ACTION_DELTAS):
            raise ValueError(f"Invalid action: {action}")

        previous_observation = self.observation()
        target_before_step = self.goal if self.has_key else self.key
        distance_before = self._path_distance(self.position, target_before_step)
        row_delta, col_delta = ACTION_DELTAS[action]
        candidate = (
            self.position[0] + row_delta,
            self.position[1] + col_delta,
        )
        wall_hit = not self._walkable(candidate)
        if not wall_hit:
            self.position = candidate
        self.steps += 1

        picked_key = not self.has_key and self.position == self.key
        if picked_key:
            self.has_key = True
        success = self.has_key and self.position == self.goal
        self.done = success or self.steps >= self.config.max_steps
        progress = (
            distance_before - self._path_distance(self.position, target_before_step)
        ) / (2 * (self.size - 1))
        features = np.asarray(
            (
                progress,
                float(picked_key),
                float(success),
                float(self.position in self.hazards),
                float(wall_hit),
                1.0,
            ),
            dtype=np.float32,
        )
        # The oracle stays outside policy and reward-model training.
        oracle_reward = float(
            features @ np.asarray(self.config.oracle_weights, dtype=np.float32)
        )
        return StepResult(
            observation=self.observation(),
            features=features,
            context=previous_observation.scalars[:3].copy(),
            oracle_reward=oracle_reward,
            done=self.done,
            success=success,
        )

    def expert_action(self, epsilon: float = 0.0) -> int:
        """Choose a shortest safe route, with optional exploration noise."""
        if self._rng.random() < epsilon:
            return self.random_action()
        target = self.goal if self.has_key else self.key
        for avoid_hazards in (True, False):
            queue: deque[tuple[tuple[int, int], int | None]] = deque(
                [(self.position, None)]
            )
            visited = {self.position}
            while queue:
                position, first_action = queue.popleft()
                if position == target and first_action is not None:
                    return first_action
                for action, (row_delta, col_delta) in enumerate(ACTION_DELTAS):
                    next_position = (
                        position[0] + row_delta,
                        position[1] + col_delta,
                    )
                    if (
                        next_position in visited
                        or not self._walkable(next_position)
                        or (avoid_hazards and next_position in self.hazards)
                    ):
                        continue
                    visited.add(next_position)
                    queue.append(
                        (next_position, action if first_action is None else first_action)
                    )
        return self.random_action()

    def random_action(self) -> int:
        """Sample an action from the environment's reproducible RNG."""
        return int(self._rng.integers(len(ACTION_DELTAS)))

    def _walkable(self, position: tuple[int, int]) -> bool:
        row, col = position
        return (
            0 <= row < self.size
            and 0 <= col < self.size
            and position not in self.walls
        )

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _path_distance(
        self, start: tuple[int, int], target: tuple[int, int]
    ) -> int:
        """Measure route progress through free cells instead of through walls."""
        queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            position, distance = queue.popleft()
            if position == target:
                return distance
            for row_delta, col_delta in ACTION_DELTAS:
                next_position = (
                    position[0] + row_delta,
                    position[1] + col_delta,
                )
                if (
                    next_position in visited
                    or not self._walkable(next_position)
                    or next_position in self.hazards
                ):
                    continue
                visited.add(next_position)
                queue.append((next_position, distance + 1))
        return self._distance(start, target)
