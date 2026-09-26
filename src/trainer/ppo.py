"""On-policy PPO with generalized advantage estimation and a learned reward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from src.config.demo import PPOConfig
from src.envs.grid_robot import GridRobotEnv, Observation
from src.network.actor_critic import ActorCritic
from src.network.reward_model import RewardEnsemble
from src.trainer.reference_states import ReferenceStateSampler


@dataclass
class Rollout:
    """Dense on-policy experience with model-predicted transition rewards."""

    grids: torch.Tensor
    scalars: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_value: float
    completed_episodes: int
    completed_successes: int
    reference_episodes: int


def _observation_tensors(
    observation: Observation, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(
            observation.grid[None], dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            observation.scalars[None], dtype=torch.float32, device=device
        ),
    )


def collect_rollout(
    env: GridRobotEnv,
    policy: ActorCritic,
    reward_model: RewardEnsemble,
    steps: int,
    seed: int,
    device: torch.device,
    reference_sampler: ReferenceStateSampler | None = None,
) -> Rollout:
    """Collect policy actions, then label transitions with the learned reward."""
    if reference_sampler is None:
        observation = env.reset(seed=seed)
        reference_index = None
    else:
        observation, reference_index = reference_sampler.reset_episode(
            env, seed=seed
        )
    grids: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    features: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    dones: list[float] = []
    completed_episodes = 0
    completed_successes = 0
    reference_episodes = 0

    policy.eval()
    reward_model.eval()
    for _ in range(steps):
        grid_tensor, scalar_tensor = _observation_tensors(observation, device)
        action, log_prob, value = policy.select_action(
            grid_tensor, scalar_tensor
        )
        transition = env.step(action)
        grids.append(observation.grid)
        scalars.append(observation.scalars)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        features.append(transition.features)
        contexts.append(transition.context)
        dones.append(float(transition.done))
        observation = transition.observation
        if transition.done:
            completed_episodes += 1
            completed_successes += int(transition.success)
            reference_episodes += int(reference_index is not None)
            if reference_sampler is None:
                observation = env.reset()
            else:
                reference_sampler.observe_outcome(
                    reference_index, transition.success
                )
                observation, reference_index = reference_sampler.reset_episode(env)

    # A truncated episode needs a final value bootstrap; a terminal one does not.
    if dones[-1]:
        last_value = 0.0
    else:
        grid_tensor, scalar_tensor = _observation_tensors(observation, device)
        with torch.no_grad():
            _, final_value = policy(grid_tensor, scalar_tensor)
        last_value = float(final_value.item())
    feature_tensor = torch.as_tensor(
        np.stack(features), dtype=torch.float32, device=device
    )
    context_tensor = torch.as_tensor(
        np.stack(contexts), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        predicted_rewards = reward_model.predict_step(
            feature_tensor, context_tensor
        )
    return Rollout(
        grids=torch.as_tensor(np.stack(grids), device=device),
        scalars=torch.as_tensor(np.stack(scalars), device=device),
        actions=torch.as_tensor(actions, dtype=torch.long, device=device),
        old_log_probs=torch.as_tensor(log_probs, device=device),
        old_values=torch.as_tensor(values, device=device),
        rewards=predicted_rewards,
        dones=torch.as_tensor(dones, device=device),
        last_value=last_value,
        completed_episodes=completed_episodes,
        completed_successes=completed_successes,
        reference_episodes=reference_episodes,
    )


def generalized_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap only across transitions inside the same episode."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device)
    next_value = torch.as_tensor(last_value, device=rewards.device)
    for index in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages + values


def update_policy(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: PPOConfig,
) -> dict[str, float]:
    """Apply clipped PPO policy and value updates to one frozen rollout."""
    advantages, returns = generalized_advantages(
        rollout.rewards,
        rollout.old_values,
        rollout.dones,
        rollout.last_value,
        config.gamma,
        config.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1.0e-8
    )
    sample_count = len(rollout.actions)
    aggregate: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }
    policy.train()
    early_stop = False
    for _ in range(config.epochs):
        order = torch.randperm(sample_count, device=rollout.actions.device)
        for indices in order.split(config.minibatch_size):
            logits, values = policy(
                rollout.grids[indices], rollout.scalars[indices]
            )
            distribution = Categorical(logits=logits)
            log_prob = distribution.log_prob(rollout.actions[indices])
            log_ratio = log_prob - rollout.old_log_probs[indices]
            ratio = log_ratio.exp()
            with torch.no_grad():
                current_kl = float(((ratio - 1.0) - log_ratio).mean())
            if current_kl > config.target_kl:
                early_stop = True
                break
            surrogate = torch.minimum(
                ratio * advantages[indices],
                ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range)
                * advantages[indices],
            )
            policy_loss = -surrogate.mean()
            value_loss = 0.5 * (values - returns[indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), config.max_grad_norm
            )
            optimizer.step()
            with torch.no_grad():
                aggregate["policy_loss"].append(float(policy_loss))
                aggregate["value_loss"].append(float(value_loss))
                aggregate["entropy"].append(float(entropy))
                aggregate["approx_kl"].append(
                    current_kl
                )
                aggregate["clip_fraction"].append(
                    float(((ratio - 1.0).abs() > config.clip_range).float().mean())
                )
        if early_stop:
            break
    old_variance = float(returns.var(unbiased=False))
    explained_variance = (
        1.0
        - float((returns - rollout.old_values).var(unbiased=False))
        / (old_variance + 1.0e-8)
    )
    return {
        **{name: float(np.mean(values)) for name, values in aggregate.items()},
        "explained_variance": explained_variance,
        "rollout_success_rate": (
            rollout.completed_successes / max(rollout.completed_episodes, 1)
        ),
        "rollout_episodes": float(rollout.completed_episodes),
        "reference_episode_fraction": (
            rollout.reference_episodes / max(rollout.completed_episodes, 1)
        ),
        "learned_reward_mean": float(rollout.rewards.mean()),
        "early_stop": float(early_stop),
    }
