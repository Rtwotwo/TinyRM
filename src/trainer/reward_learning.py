"""Preference optimization for the structured reward ensemble."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from src.network.reward_model import RewardEnsemble
from src.trainer.preferences import Preference, Trajectory, pad_trajectories


def train_reward_ensemble(
    ensemble: RewardEnsemble,
    trajectories: list[Trajectory],
    preferences: list[Preference],
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Train members on independent bootstrap samples of pairwise labels."""
    features, contexts, mask = pad_trajectories(trajectories, device)
    left = torch.as_tensor([pair.left for pair in preferences], device=device)
    right = torch.as_tensor([pair.right for pair in preferences], device=device)
    labels = torch.as_tensor(
        [pair.label for pair in preferences], dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(seed)
    final_losses: list[float] = []
    final_accuracies: list[float] = []
    for member in ensemble.members:
        optimizer = torch.optim.Adam(member.parameters(), lr=2.0e-3)
        for _ in range(epochs):
            sampled = torch.as_tensor(
                rng.integers(len(preferences), size=len(preferences)),
                dtype=torch.long,
                device=device,
            )
            returns = (member(features, contexts) * mask).sum(dim=-1)
            logits = returns[left[sampled]] - returns[right[sampled]]
            loss = F.binary_cross_entropy_with_logits(logits, labels[sampled])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), 5.0)
            optimizer.step()
        with torch.no_grad():
            returns = (member(features, contexts) * mask).sum(dim=-1)
            logits = returns[left] - returns[right]
            final_losses.append(
                float(F.binary_cross_entropy_with_logits(logits, labels))
            )
            final_accuracies.append(
                float(((logits > 0).float() == labels).float().mean())
            )
    return {
        "reward_preference_loss": float(np.mean(final_losses)),
        "reward_preference_accuracy": float(np.mean(final_accuracies)),
    }
