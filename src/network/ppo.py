"""PPO policy architecture used for G1 dance motion tracking."""

from collections.abc import Sequence
from typing import Any


def build_g1_dance_policy_kwargs(
    hidden_sizes: Sequence[int], initial_log_std: float = -0.7
) -> dict[str, Any]:
    """Build matching actor and critic MLP settings for Stable-Baselines3 PPO."""
    import torch

    network_sizes = list(hidden_sizes)
    if not network_sizes or any(width <= 0 for width in network_sizes):
        raise ValueError("hidden_sizes must contain positive layer widths.")

    return {
        "net_arch": {
            "pi": network_sizes,
            "vf": network_sizes,
        },
        "activation_fn": torch.nn.Tanh,
        "log_std_init": float(initial_log_std),
    }