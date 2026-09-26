"""Validated configuration for the grid robot self-improvement demo."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvironmentConfig:
    """Geometry and hidden evaluation objective for the grid robot."""

    size: int = 7
    max_steps: int = 28
    # These weights label synthetic preferences and evaluate policies.
    # The policy and learned reward model never receive them directly.
    oracle_weights: tuple[float, ...] = (
        2.0, 1.5, 5.0, -0.8, -0.15, -0.025
    )

    def __post_init__(self) -> None:
        if self.size != 7:
            raise ValueError("The demonstration map is defined for size 7.")
        if self.max_steps < 16:
            raise ValueError("max_steps must permit the expert route.")
        if len(self.oracle_weights) != 6:
            raise ValueError("The oracle needs one weight per reward feature.")


@dataclass(frozen=True)
class ModelConfig:
    """Network capacity and structured reward ensemble size."""

    hidden_dim: int = 96
    reward_members: int = 3
    reward_context_dim: int = 32

    def __post_init__(self) -> None:
        if self.hidden_dim < 32 or self.reward_members < 2:
            raise ValueError("Use a nontrivial encoder and at least two reward members.")


@dataclass(frozen=True)
class PPOConfig:
    """On-policy optimization settings for the actor and value function."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 8.0e-5
    epochs: int = 4
    minibatch_size: int = 128
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.02


@dataclass(frozen=True)
class RSIConfig:
    """Bounded policy and reward-model improvement loop."""

    rounds: int = 3
    rollout_steps_per_round: int = 1536
    initial_trajectories: int = 64
    initial_preferences: int = 128
    new_trajectories_per_round: int = 24
    new_preferences_per_round: int = 48
    reward_epochs: int = 35
    evaluation_episodes: int = 16
    query_pool_size: int = 80
    warm_start_episodes: int = 48
    warm_start_epochs: int = 5

    def __post_init__(self) -> None:
        if min(
            self.rounds,
            self.rollout_steps_per_round,
            self.initial_trajectories,
            self.initial_preferences,
            self.new_trajectories_per_round,
            self.new_preferences_per_round,
            self.reward_epochs,
            self.evaluation_episodes,
            self.warm_start_episodes,
            self.warm_start_epochs,
        ) <= 0:
            raise ValueError("All RSI loop sizes must be positive.")


@dataclass(frozen=True)
class DemoConfig:
    """Complete experiment configuration."""

    seed: int = 42
    device: str = "cpu"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rsi: RSIConfig = field(default_factory=RSIConfig)
