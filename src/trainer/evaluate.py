"""Reload a saved policy and create an independent evaluation report."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

from src.config.demo import (
    DemoConfig,
    EnvironmentConfig,
    ModelConfig,
    PPOConfig,
    RSIConfig,
)
from src.network.actor_critic import ActorCritic
from src.trainer.rsi import evaluate_policy


def _load_config(path: Path) -> DemoConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return DemoConfig(
        seed=raw["seed"],
        device=raw["device"],
        environment=EnvironmentConfig(**raw["environment"]),
        model=ModelConfig(**raw["model"]),
        ppo=PPOConfig(**raw["ppo"]),
        rsi=RSIConfig(**raw["rsi"]),
    )


def main() -> None:
    """Evaluate a checkpoint without creating files outside its experiment run."""
    parser = argparse.ArgumentParser(description="Evaluate a saved grid robot policy.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    if arguments.episodes <= 0:
        parser.error("--episodes must be positive")
    checkpoint = arguments.checkpoint.resolve()
    run_dir = checkpoint.parent
    project_exp = Path(__file__).resolve().parents[2] / "exp"
    if not checkpoint.is_relative_to(project_exp):
        parser.error("The checkpoint must be inside this project's exp/ directory.")
    config = _load_config(run_dir / "config.json")
    config = replace(
        config,
        device=arguments.device,
        rsi=replace(config.rsi, evaluation_episodes=arguments.episodes),
    )
    device = torch.device(config.device)
    policy = ActorCritic(config.model.hidden_dim).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(saved["actor_critic"])
    policy.eval()
    metrics, _ = evaluate_policy(
        policy, config, device, first_seed=arguments.seed
    )
    report_path = (
        run_dir
        / f"evaluation_{checkpoint.stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    report_path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "episodes": arguments.episodes,
                "seed": arguments.seed,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Evaluation: {report_path}")


if __name__ == "__main__":
    main()
