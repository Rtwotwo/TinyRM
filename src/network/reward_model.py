"""Interpretable, context-conditioned reward models trained from preferences."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StructuredRewardModel(nn.Module):
    """Assign signed contributions to six observable transition events.

    Positive events cannot become penalties, and safety violations cannot
    become bonuses. The context network adjusts each magnitude by at most 25%.
    """

    def __init__(self, context_width: int) -> None:
        super().__init__()
        self.raw_magnitudes = nn.Parameter(
            torch.tensor((1.8, 1.3, 4.5, 0.7, 0.2, 0.03)).log()
        )
        self.context_network = nn.Sequential(
            nn.Linear(3, context_width),
            nn.SiLU(),
            nn.Linear(context_width, context_width),
            nn.SiLU(),
            nn.Linear(context_width, 6),
        )
        # Starting with zero modulation makes the initial decomposition legible.
        nn.init.zeros_(self.context_network[-1].weight)
        nn.init.zeros_(self.context_network[-1].bias)
        self.register_buffer(
            "signs", torch.tensor((1.0, 1.0, 1.0, -1.0, -1.0, -1.0))
        )

    def components(
        self, features: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        magnitudes = F.softplus(self.raw_magnitudes)
        modulation = 1.0 + 0.25 * torch.tanh(self.context_network(context))
        return features * self.signs * magnitudes * modulation

    def forward(
        self, features: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.components(features, context).sum(dim=-1)


class RewardEnsemble(nn.Module):
    """Estimate preference uncertainty with independently trained members."""

    def __init__(self, member_count: int, context_width: int) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            StructuredRewardModel(context_width) for _ in range(member_count)
        )

    def member_returns(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [(member(features, context) * mask).sum(dim=-1) for member in self.members],
            dim=0,
        )

    @torch.no_grad()
    def predict_step(
        self, features: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(
            [member(features, context) for member in self.members], dim=0
        ).mean(dim=0)

    @torch.no_grad()
    def mean_components(
        self, features: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(
            [member.components(features, context) for member in self.members], dim=0
        ).mean(dim=0)
