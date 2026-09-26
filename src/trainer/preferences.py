"""Trajectory storage and uncertainty-driven preference queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.envs.grid_robot import GridRobotEnv
from src.network.actor_critic import ActorCritic
from src.network.reward_model import RewardEnsemble


@dataclass(frozen=True)
class Trajectory:
    """One complete rollout, retaining oracle scores only for the judge."""

    features: np.ndarray
    contexts: np.ndarray
    oracle_return: float
    success: bool
    positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Preference:
    """An externally labeled comparison between two archive trajectories."""

    left: int
    right: int
    label: float


def collect_trajectory(
    env: GridRobotEnv,
    seed: int,
    device: torch.device,
    policy: ActorCritic | None = None,
    mode: str = "policy",
    deterministic: bool = False,
) -> Trajectory:
    """Collect one episode without exposing the oracle to the acting policy."""
    observation = env.reset(seed=seed)
    features: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    positions = [env.position]
    oracle_return = 0.0
    while True:
        if mode == "expert":
            action = env.expert_action(epsilon=0.15)
        elif mode == "random":
            action = env.random_action()
        elif mode == "policy" and policy is not None:
            grid = torch.as_tensor(
                observation.grid[None], dtype=torch.float32, device=device
            )
            scalars = torch.as_tensor(
                observation.scalars[None], dtype=torch.float32, device=device
            )
            action, _, _ = policy.select_action(
                grid, scalars, deterministic=deterministic
            )
        else:
            raise ValueError(f"Unsupported trajectory mode: {mode}")
        step = env.step(action)
        features.append(step.features)
        contexts.append(step.context)
        positions.append(env.position)
        oracle_return += step.oracle_reward
        observation = step.observation
        if step.done:
            return Trajectory(
                features=np.stack(features),
                contexts=np.stack(contexts),
                oracle_return=oracle_return,
                success=step.success,
                positions=tuple(positions),
            )


def judge_pair(
    trajectories: list[Trajectory], left: int, right: int
) -> Preference | None:
    """Return a synthetic preference from the hidden task objective."""
    difference = (
        trajectories[left].oracle_return - trajectories[right].oracle_return
    )
    if abs(difference) < 1.0e-5:
        return None
    return Preference(left, right, float(difference > 0.0))


def sample_initial_preferences(
    trajectories: list[Trajectory], count: int, rng: np.random.Generator
) -> list[Preference]:
    """Build a diverse initial preference set from expert and random rollouts."""
    preferences: list[Preference] = []
    attempts = 0
    while len(preferences) < count and attempts < 20 * count:
        left, right = rng.choice(len(trajectories), size=2, replace=False)
        pair = judge_pair(trajectories, int(left), int(right))
        if pair is not None:
            preferences.append(pair)
        attempts += 1
    if len(preferences) < count:
        raise RuntimeError("Not enough distinct trajectories for preference labels.")
    return preferences


def pad_trajectories(
    trajectories: list[Trajectory], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length episodes while keeping an explicit validity mask."""
    longest = max(len(trajectory.features) for trajectory in trajectories)
    features = np.zeros((len(trajectories), longest, 6), dtype=np.float32)
    contexts = np.zeros((len(trajectories), longest, 3), dtype=np.float32)
    mask = np.zeros((len(trajectories), longest), dtype=np.float32)
    for index, trajectory in enumerate(trajectories):
        length = len(trajectory.features)
        features[index, :length] = trajectory.features
        contexts[index, :length] = trajectory.contexts
        mask[index, :length] = 1.0
    return (
        torch.as_tensor(features, device=device),
        torch.as_tensor(contexts, device=device),
        torch.as_tensor(mask, device=device),
    )


@torch.no_grad()
def query_uncertain_pairs(
    ensemble: RewardEnsemble,
    trajectories: list[Trajectory],
    old_count: int,
    count: int,
    pool_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> list[Preference]:
    """Label candidate pairs with high ensemble disagreement and low margin."""
    features, contexts, mask = pad_trajectories(trajectories, device)
    member_returns = ensemble.member_returns(features, contexts, mask).cpu().numpy()
    scored: list[tuple[float, int, int]] = []
    for _ in range(max(pool_size, count * 4)):
        left = int(rng.integers(old_count, len(trajectories)))
        right = int(rng.integers(len(trajectories)))
        if left == right:
            continue
        differences = member_returns[:, left] - member_returns[:, right]
        margin = abs(float(differences.mean()))
        uncertainty = float(
            (0.5 + differences.std()) / (1.0 + margin)
        )
        scored.append((uncertainty, left, right))
    scored.sort(reverse=True)
    preferences: list[Preference] = []
    seen: set[tuple[int, int]] = set()
    for _, left, right in scored:
        key = (min(left, right), max(left, right))
        if key in seen:
            continue
        seen.add(key)
        pair = judge_pair(trajectories, left, right)
        if pair is not None:
            preferences.append(pair)
        if len(preferences) == count:
            break
    if len(preferences) < count:
        preferences.extend(
            sample_initial_preferences(trajectories, count - len(preferences), rng)
        )
    return preferences
