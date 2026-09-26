"""Command-line entry point for the bounded self-improvement experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.config.demo import DemoConfig
from src.trainer.rsi import run_experiment


def main() -> None:
    """Run the demo and keep every generated artifact under exp/."""
    parser = argparse.ArgumentParser(
        description="Structured reward and value learning with bounded RSI."
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start-epochs", type=int, default=5)
    parser.add_argument("--warm-start-episodes", type=int, default=48)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    config = DemoConfig(
        seed=arguments.seed,
        device=arguments.device,
        rsi=replace(
            DemoConfig().rsi,
            rounds=arguments.rounds,
            rollout_steps_per_round=arguments.rollout_steps,
            warm_start_epochs=arguments.warm_start_epochs,
            warm_start_episodes=arguments.warm_start_episodes,
        ),
    )
    project_root = Path(__file__).resolve().parents[2]
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "exp" / "rsi_demo" / run_name
    report = run_experiment(config, output_dir)
    print(f"Artifacts: {output_dir}")
    print(f"Validation: {report['final_champion']}")
    print(f"Untouched test: {report['untouched_test']['final']}")


if __name__ == "__main__":
    main()
