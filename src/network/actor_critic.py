"""Separate spatial policy and value networks for discrete robot control."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    """Refine a hidden state without discarding its original representation."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class SpatialEncoder(nn.Module):
    """Fuse a spatial map with scalar task and time information."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(width + 32, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            ResidualBlock(width),
            ResidualBlock(width),
        )

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        spatial = self.map_encoder(grid)
        state = self.state_encoder(scalars)
        return self.fusion(torch.cat((spatial, state), dim=-1))


class ActorCritic(nn.Module):
    """Use independent encoders to avoid policy and value gradient interference."""

    def __init__(self, width: int, action_count: int = 4) -> None:
        super().__init__()
        self.actor_encoder = SpatialEncoder(width)
        self.critic_encoder = SpatialEncoder(width)
        self.actor_head = nn.Linear(width, action_count)
        self.value_head = nn.Linear(width, 1)

    def forward(
        self, grid: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor_head(self.actor_encoder(grid, scalars))
        value = self.value_head(self.critic_encoder(grid, scalars)).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def select_action(
        self, grid: torch.Tensor, scalars: torch.Tensor, deterministic: bool = False
    ) -> tuple[int, float, float]:
        logits, value = self(grid, scalars)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
        )
