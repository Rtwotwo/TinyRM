"""Bounded recursive self-improvement with an external acceptance gate."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from src.config.demo import DemoConfig
from src.envs.grid_robot import GridRobotEnv, REWARD_FEATURE_NAMES
from src.network.actor_critic import ActorCritic
from src.network.reward_model import RewardEnsemble
from src.trainer.ppo import collect_rollout, update_policy
from src.trainer.preferences import (
    Preference,
    Trajectory,
    collect_trajectory,
    judge_pair,
    query_uncertain_pairs,
    sample_initial_preferences,
)
from src.trainer.reward_learning import train_reward_ensemble
from src.trainer.reference_states import ReferenceStateSampler


def warm_start_policy(
    policy: ActorCritic, config: DemoConfig, device: torch.device
) -> float:
    """Learn useful navigation behavior from an expert without reward labels."""
    env = GridRobotEnv(config.environment)
    grids: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    actions: list[int] = []
    for episode in range(config.rsi.warm_start_episodes):
        observation = env.reset(seed=config.seed + 1000 + episode)
        while True:
            grids.append(observation.grid)
            scalars.append(observation.scalars)
            action = env.expert_action(epsilon=0.05)
            actions.append(action)
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break
    grid_tensor = torch.as_tensor(np.stack(grids), device=device)
    scalar_tensor = torch.as_tensor(np.stack(scalars), device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    final_loss = 0.0
    policy.train()
    for _ in range(config.rsi.warm_start_epochs):
        order = torch.randperm(len(actions), device=device)
        for indices in order.split(128):
            logits, _ = policy(grid_tensor[indices], scalar_tensor[indices])
            loss = F.cross_entropy(logits, action_tensor[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach())
    return final_loss


def evaluate_policy(
    policy: ActorCritic,
    config: DemoConfig,
    device: torch.device,
    first_seed: int,
) -> tuple[dict[str, float], Trajectory]:
    """Score stochastic PPO behavior on fixed held-out starts and random seeds."""
    env = GridRobotEnv(config.environment)
    episodes: list[Trajectory] = []
    deterministic_episodes: list[Trajectory] = []
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    for episode in range(config.rsi.evaluation_episodes):
        episode_seed = first_seed + episode
        # Fork the RNG so evaluations do not change subsequent training samples.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(episode_seed)
            episodes.append(
                collect_trajectory(
                    env,
                    seed=episode_seed,
                    device=device,
                    policy=policy,
                )
            )
        deterministic_episodes.append(
            collect_trajectory(
                env,
                seed=episode_seed,
                device=device,
                policy=policy,
                deterministic=True,
            )
        )
    summary = {
        "success_rate": float(np.mean([episode.success for episode in episodes])),
        "oracle_return": float(
            np.mean([episode.oracle_return for episode in episodes])
        ),
        "mean_episode_length": float(
            np.mean([len(episode.features) for episode in episodes])
        ),
        "hazard_contacts": float(
            np.mean([episode.features[:, 3].sum() for episode in episodes])
        ),
        "wall_collisions": float(
            np.mean([episode.features[:, 4].sum() for episode in episodes])
        ),
        "deterministic_success_rate": float(
            np.mean([episode.success for episode in deterministic_episodes])
        ),
        "deterministic_oracle_return": float(
            np.mean([episode.oracle_return for episode in deterministic_episodes])
        ),
    }
    representative = max(
        episodes,
        key=lambda episode: (episode.success, episode.oracle_return),
    )
    return summary, representative


def heldout_preference_accuracy(
    ensemble: RewardEnsemble,
    trajectories: list[Trajectory],
    pairs: list[Preference],
    device: torch.device,
) -> float:
    """Measure whether learned rankings transfer to unseen trajectories."""
    from src.trainer.preferences import pad_trajectories

    features, contexts, mask = pad_trajectories(trajectories, device)
    with torch.no_grad():
        returns = ensemble.member_returns(features, contexts, mask).mean(dim=0)
    correct = [
        float(
            (returns[pair.left] > returns[pair.right]).item()
            == bool(pair.label)
        )
        for pair in pairs
    ]
    return float(np.mean(correct))


def _initial_trajectory_archive(
    policy: ActorCritic, config: DemoConfig, device: torch.device
) -> list[Trajectory]:
    env = GridRobotEnv(config.environment)
    modes = ("expert", "random", "policy")
    return [
        collect_trajectory(
            env,
            seed=config.seed + 2000 + index,
            device=device,
            policy=policy,
            mode=modes[index % len(modes)],
        )
        for index in range(config.rsi.initial_trajectories)
    ]


def _heldout_preferences(
    policy: ActorCritic, config: DemoConfig, device: torch.device
) -> tuple[list[Trajectory], list[Preference]]:
    env = GridRobotEnv(config.environment)
    modes = ("expert", "random", "policy")
    trajectories = [
        collect_trajectory(
            env,
            seed=config.seed + 20000 + index,
            device=device,
            policy=policy,
            mode=modes[index % len(modes)],
        )
        for index in range(24)
    ]
    rng = np.random.default_rng(config.seed + 30000)
    pairs = sample_initial_preferences(trajectories, 48, rng)
    return trajectories, pairs


def _accept(
    candidate: dict[str, float], champion: dict[str, float]
) -> bool:
    """Require nondecreasing success and a measurable objective improvement."""
    if candidate["success_rate"] > champion["success_rate"] + 1.0e-9:
        return True
    return (
        candidate["success_rate"] >= champion["success_rate"] - 1.0e-9
        and candidate["oracle_return"] > champion["oracle_return"] + 1.0e-3
    )


def _plot_results(
    output_dir: Path,
    records: list[dict[str, object]],
    trajectory: Trajectory,
    config: DemoConfig,
) -> None:
    """Create a compact learning curve and policy path as experiment artifacts."""
    rounds = [int(record["round"]) for record in records]
    champion_rates = [
        float(record["champion"]["success_rate"]) for record in records
    ]
    candidate_rates = [
        float(record["candidate"]["success_rate"]) for record in records
    ]
    champion_returns = [
        float(record["champion"]["oracle_return"]) for record in records
    ]
    candidate_returns = [
        float(record["candidate"]["oracle_return"]) for record in records
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(rounds, champion_rates, marker="o", label="Accepted policy")
    axes[0].plot(
        rounds, candidate_rates, marker="x", linestyle="--", label="Candidate"
    )
    axes[0].set(title="Success rate", xlabel="RSI round", ylim=(-0.05, 1.05))
    axes[0].legend()
    axes[1].plot(rounds, champion_returns, marker="o", label="Accepted policy")
    axes[1].plot(
        rounds, candidate_returns, marker="x", linestyle="--", label="Candidate"
    )
    axes[1].set(title="Held-out oracle return", xlabel="RSI round")
    axes[1].legend()

    env = GridRobotEnv(config.environment)
    map_image = np.ones((env.size, env.size, 3), dtype=np.float32)
    for row, col in env.walls:
        map_image[row, col] = (0.2, 0.2, 0.2)
    for row, col in env.hazards:
        map_image[row, col] = (0.95, 0.55, 0.25)
    map_image[env.key] = (0.98, 0.85, 0.1)
    map_image[env.goal] = (0.2, 0.75, 0.25)
    axes[2].imshow(map_image)
    positions = np.asarray(trajectory.positions)
    axes[2].plot(positions[:, 1], positions[:, 0], "b.-", linewidth=1.5)
    axes[2].scatter(positions[0, 1], positions[0, 0], c="cyan", s=80)
    axes[2].set(
        title="Final held-out trajectory",
        xticks=range(env.size),
        yticks=range(env.size),
    )
    fig.tight_layout()
    fig.savefig(output_dir / "learning_and_trajectory.png", dpi=160)
    plt.close(fig)


def run_experiment(
    config: DemoConfig, output_dir: Path
) -> dict[str, object]:
    """Improve policy and reward models under a fixed external judge."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    champion = ActorCritic(config.model.hidden_dim).to(device)
    reward_model = RewardEnsemble(
        config.model.reward_members, config.model.reward_context_dim
    ).to(device)
    warm_start_loss = warm_start_policy(champion, config, device)
    archive = _initial_trajectory_archive(champion, config, device)
    preferences = sample_initial_preferences(
        archive, config.rsi.initial_preferences, rng
    )
    holdout_trajectories, holdout_pairs = _heldout_preferences(
        champion, config, device
    )
    reward_stats = train_reward_ensemble(
        reward_model,
        archive,
        preferences,
        config.rsi.reward_epochs,
        config.seed,
        device,
    )
    reward_stats["heldout_preference_accuracy"] = heldout_preference_accuracy(
        reward_model, holdout_trajectories, holdout_pairs, device
    )
    baseline, final_trajectory = evaluate_policy(
        champion, config, device, first_seed=config.seed + 50000
    )
    baseline_test, _ = evaluate_policy(
        champion, config, device, first_seed=config.seed + 90000
    )
    records: list[dict[str, object]] = [
        {
            "round": 0,
            "accepted": True,
            "champion": baseline,
            "candidate": baseline,
            "reward_model": reward_stats,
            "behavior_cloning_loss": warm_start_loss,
            "environment_steps": 0,
        }
    ]
    (output_dir / "metrics.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "actor_critic": champion.state_dict(),
            "reward_ensemble": reward_model.state_dict(),
            "round": 0,
        },
        output_dir / "champion_round_0.pt",
    )

    total_steps = 0
    reference_sampler = ReferenceStateSampler(
        config.environment, seed=config.seed + 40000
    )
    for round_index in range(1, config.rsi.rounds + 1):
        candidate = copy.deepcopy(champion)
        optimizer = torch.optim.Adam(
            candidate.parameters(), lr=config.ppo.learning_rate
        )
        update_metrics: list[dict[str, float]] = []
        steps_remaining = config.rsi.rollout_steps_per_round
        rollout_index = 0
        while steps_remaining:
            step_count = min(256, steps_remaining)
            rollout = collect_rollout(
                GridRobotEnv(config.environment),
                candidate,
                reward_model,
                step_count,
                config.seed + 60000 + round_index * 100 + rollout_index,
                device,
                reference_sampler,
            )
            update_metrics.append(
                update_policy(candidate, optimizer, rollout, config.ppo)
            )
            steps_remaining -= step_count
            total_steps += step_count
            rollout_index += 1

        candidate_score, candidate_trajectory = evaluate_policy(
            candidate, config, device, first_seed=config.seed + 50000
        )
        previous_score = records[-1]["champion"]
        accepted = _accept(candidate_score, previous_score)
        if accepted:
            champion = candidate
            final_trajectory = candidate_trajectory
            champion_score = candidate_score
        else:
            champion_score = previous_score

        old_count = len(archive)
        env = GridRobotEnv(config.environment)
        for index in range(config.rsi.new_trajectories_per_round):
            archive.append(
                collect_trajectory(
                    env,
                    seed=config.seed + 70000 + round_index * 1000 + index,
                    device=device,
                    policy=candidate,
                )
            )
        new_preferences = query_uncertain_pairs(
            reward_model,
            archive,
            old_count,
            config.rsi.new_preferences_per_round,
            config.rsi.query_pool_size,
            rng,
            device,
        )
        preferences.extend(new_preferences)
        reward_stats = train_reward_ensemble(
            reward_model,
            archive,
            preferences,
            config.rsi.reward_epochs,
            config.seed + round_index,
            device,
        )
        reward_stats["heldout_preference_accuracy"] = (
            heldout_preference_accuracy(
                reward_model, holdout_trajectories, holdout_pairs, device
            )
        )
        ppo_metrics = {
            key: float(np.mean([metrics[key] for metrics in update_metrics]))
            for key in update_metrics[0]
        }
        record: dict[str, object] = {
            "round": round_index,
            "accepted": accepted,
            "champion": champion_score,
            "candidate": candidate_score,
            "ppo": ppo_metrics,
            "reward_model": reward_stats,
            "environment_steps": total_steps,
            "trajectory_count": len(archive),
            "preference_count": len(preferences),
        }
        records.append(record)
        torch.save(
            {
                "actor_critic": candidate.state_dict(),
                "reward_ensemble": reward_model.state_dict(),
                "round": round_index,
                "accepted": accepted,
            },
            output_dir / f"candidate_round_{round_index}.pt",
        )
        torch.save(
            {
                "actor_critic": champion.state_dict(),
                "reward_ensemble": reward_model.state_dict(),
                "round": round_index,
            },
            output_dir / f"champion_round_{round_index}.pt",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        print(
            f"Round {round_index}/{config.rsi.rounds}: "
            f"accepted={accepted}, success={champion_score['success_rate']:.3f}, "
            f"return={champion_score['oracle_return']:.3f}, "
            f"preference_accuracy="
            f"{reward_stats['heldout_preference_accuracy']:.3f}",
            flush=True,
        )

    _plot_results(output_dir, records, final_trajectory, config)
    final_test, _ = evaluate_policy(
        champion, config, device, first_seed=config.seed + 90000
    )
    example_features = torch.eye(6, device=device)
    example_context = torch.tensor(
        [[0.0, 1.0, 0.5]] * 6, device=device
    )
    with torch.no_grad():
        component_values = reward_model.mean_components(
            example_features, example_context
        ).sum(dim=0)
    report = {
        "experiment": "bounded_recursive_self_improvement",
        "rounds": records,
        "final_champion": records[-1]["champion"],
        "untouched_test": {
            "baseline": baseline_test,
            "final": final_test,
        },
        "structured_reward_components": {
            name: float(value)
            for name, value in zip(REWARD_FEATURE_NAMES, component_values)
        },
        "artifacts": {
            "metrics": "metrics.json",
            "figure": "learning_and_trajectory.png",
            "last_checkpoint": f"champion_round_{config.rsi.rounds}.pt",
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
