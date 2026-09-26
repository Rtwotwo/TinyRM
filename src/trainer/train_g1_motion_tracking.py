"""Train a CUDA PPO controller to track a procedural G1 dance motion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any


from src.config.spring_dance import (
    CHECKPOINT_DIR,
    CONFIG_DIR,
    DEFAULT_G1_DANCE_PPO_CONFIG,
    DEFAULT_TRACK_MOTION,
    EVALUATION_DIR,
    G1_MODEL_XML,
    MODEL_DIR,
    MONITOR_DIR,
    TENSORBOARD_DIR,
)
from src.envs.g1_motion_tracking_env import G1MotionTrackingEnv
from src.network.ppo import build_g1_dance_policy_kwargs


def _build_dance_metrics_callback() -> Any:
    """Create a callback that logs completed-episode dance quality metrics."""
    from stable_baselines3.common.callbacks import BaseCallback

    class DanceEpisodeMetricsCallback(BaseCallback):
        """Aggregate per-step tracking metrics and log completed-episode means."""

        METRIC_KEYS = (
            "joint_rmse",
            "root_position_error",
            "root_orientation_error",
            "joint_pose_score",
            "joint_velocity_score",
            "root_position_score",
            "root_orientation_score",
            "action_rms",
            "reward_joint_pose",
            "reward_joint_velocity",
            "reward_root_position",
            "reward_root_orientation",
            "disturbance_active",
            "fell",
        )

        def __init__(self) -> None:
            super().__init__()
            self._running_totals: list[dict[str, float]] = []
            self._running_steps: list[int] = []
            self._completed_episodes: list[dict[str, float]] = []

        def _on_training_start(self) -> None:
            """Create independent accumulators for each vectorized environment."""
            from collections import defaultdict

            self._running_totals = [
                defaultdict(float) for _ in range(self.training_env.num_envs)
            ]
            self._running_steps = [0 for _ in range(self.training_env.num_envs)]

        def _on_step(self) -> bool:
            """Collect worker metrics and finalize completed episodes."""
            from collections import defaultdict

            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for env_index, info in enumerate(infos):
                totals = self._running_totals[env_index]
                self._running_steps[env_index] += 1
                for key in self.METRIC_KEYS:
                    if key in info:
                        totals[key] += float(info[key])

                if env_index < len(dones) and bool(dones[env_index]):
                    step_count = max(self._running_steps[env_index], 1)
                    episode_metrics = {
                        key: value / step_count for key, value in totals.items()
                    }
                    # Count falls as episode-level events rather than per-step averages.
                    episode_metrics["fell"] = float(totals.get("fell", 0.0) > 0.0)
                    self._completed_episodes.append(episode_metrics)
                    self._running_totals[env_index] = defaultdict(float)
                    self._running_steps[env_index] = 0
            return True

        def _on_rollout_end(self) -> None:
            """Write mean metrics for episodes completed during this rollout."""
            import numpy as np

            if not self._completed_episodes:
                return

            self.logger.record(
                "dance/episodes_completed", float(len(self._completed_episodes))
            )
            for key in self.METRIC_KEYS:
                values = [
                    episode[key]
                    for episode in self._completed_episodes
                    if key in episode
                ]
                if values:
                    self.logger.record(f"dance/{key}_mean", float(np.mean(values)))
            self.logger.record(
                "dance/fall_rate",
                float(
                    np.mean(
                        [
                            episode.get("fell", 0.0)
                            for episode in self._completed_episodes
                        ]
                    )
                ),
            )
            self._completed_episodes.clear()

    return DanceEpisodeMetricsCallback()

def parse_args() -> argparse.Namespace:
    """Parse PPO training options."""
    defaults = DEFAULT_G1_DANCE_PPO_CONFIG
    parser = argparse.ArgumentParser(
        description=(
            "Train a closed-loop PPO controller for a complex G1 dance reference. "
            "Procedural choreography is the default; GMR motions are also supported."
        )
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_TRACK_MOTION)
    parser.add_argument("--model-xml", type=Path, default=G1_MODEL_XML)
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--n-epochs", type=int, default=defaults.n_epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--target-kl", type=float, default=defaults.target_kl)
    parser.add_argument("--vf-coef", type=float, default=defaults.value_function_coefficient)
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
    parser.add_argument("--episode-seconds", type=float, default=defaults.episode_seconds)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--without-disturbances", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    """Validate the requested PPO device without silently falling back to CPU."""
    import torch
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a GPU in the current environment."
        )
    return requested


def train(args: argparse.Namespace) -> Path:
    """Train the residual controller and save all artifacts under exp/."""
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    motion_path = args.motion.expanduser().resolve()
    model_xml = args.model_xml.expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(
            f"Missing GMR reference motion: {motion_path}\n"
            "Generate a procedural reference with "
            "python -m src.trainer.generate_g1_dance_motion, or pass a GMR motion file."
        )
    if not model_xml.is_file():
        raise FileNotFoundError(f"G1 MuJoCo model not found: {model_xml}")
    if min(args.total_timesteps, args.n_envs, args.n_steps, args.batch_size, args.n_epochs) <= 0:
        raise ValueError("Training steps, environment count, batch size, and epochs must be positive.")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive.")
    if args.target_kl <= 0 or args.vf_coef <= 0:
        raise ValueError("--target-kl and --vf-coef must be positive.")
    rollout_size = args.n_envs * args.n_steps
    if args.batch_size > rollout_size or rollout_size % args.batch_size != 0:
        raise ValueError(
            "The PPO batch size must divide n_envs * n_steps; "
            f"got batch_size={args.batch_size}, rollout_size={rollout_size}."
        )

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("%Y%m%d_%H%M%S_%f")
    run_checkpoint_dir = CHECKPOINT_DIR / run_id
    run_monitor_dir = MONITOR_DIR / run_id
    for directory in (
        MODEL_DIR,
        run_checkpoint_dir,
        run_monitor_dir,
        TENSORBOARD_DIR,
        CONFIG_DIR,
        EVALUATION_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    def make_env(env_index: int) -> Any:
        """Construct one monitored MuJoCo environment inside its worker process."""
        monitor_path = run_monitor_dir / f"env_{env_index}"
        return Monitor(
            G1MotionTrackingEnv(
                motion_path=motion_path,
                model_path=model_xml,
                episode_seconds=args.episode_seconds,
                pd_kp=DEFAULT_G1_DANCE_PPO_CONFIG.pd_kp,
                pd_kd=DEFAULT_G1_DANCE_PPO_CONFIG.pd_kd,
                residual_torque_scale=DEFAULT_G1_DANCE_PPO_CONFIG.residual_torque_scale,
                action_penalty_coefficient=DEFAULT_G1_DANCE_PPO_CONFIG.action_penalty_coefficient,
                action_smoothness_penalty_coefficient=DEFAULT_G1_DANCE_PPO_CONFIG.action_smoothness_penalty_coefficient,
                observation_scales=DEFAULT_G1_DANCE_PPO_CONFIG.observation_scales,
                reset_joint_noise=DEFAULT_G1_DANCE_PPO_CONFIG.reset_joint_noise,
                root_xy_reset_noise=DEFAULT_G1_DANCE_PPO_CONFIG.root_xy_reset_noise,
                gain_randomization=DEFAULT_G1_DANCE_PPO_CONFIG.gain_randomization,
                joint_passive_damping=DEFAULT_G1_DANCE_PPO_CONFIG.joint_passive_damping,
                root_position_kp=DEFAULT_G1_DANCE_PPO_CONFIG.root_position_kp,
                root_position_kd=DEFAULT_G1_DANCE_PPO_CONFIG.root_position_kd,
                root_orientation_kp=DEFAULT_G1_DANCE_PPO_CONFIG.root_orientation_kp,
                root_orientation_kd=DEFAULT_G1_DANCE_PPO_CONFIG.root_orientation_kd,
                enable_disturbances=not args.without_disturbances,
                disturbance_force_min=DEFAULT_G1_DANCE_PPO_CONFIG.disturbance_force_min,
                disturbance_force_max=DEFAULT_G1_DANCE_PPO_CONFIG.disturbance_force_max,
                disturbance_duration_seconds=DEFAULT_G1_DANCE_PPO_CONFIG.disturbance_duration_seconds,
                disturbance_interval_min_seconds=DEFAULT_G1_DANCE_PPO_CONFIG.disturbance_interval_min_seconds,
                disturbance_interval_max_seconds=DEFAULT_G1_DANCE_PPO_CONFIG.disturbance_interval_max_seconds,
                fall_height=DEFAULT_G1_DANCE_PPO_CONFIG.fall_height,
                fall_tilt_radians=DEFAULT_G1_DANCE_PPO_CONFIG.fall_tilt_radians,
                random_start=True,
            ),
            filename=str(monitor_path),
        )

    env_fns = [lambda index=index: make_env(index) for index in range(args.n_envs)]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
        vector_env_type = "SubprocVecEnv (spawn)"
    else:
        env = DummyVecEnv(env_fns)
        vector_env_type = "DummyVecEnv"

    model_path = MODEL_DIR / f"g1_dance_ppo_{run_id}"
    config = replace(
        DEFAULT_G1_DANCE_PPO_CONFIG,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        value_function_coefficient=args.vf_coef,
        checkpoint_interval=args.checkpoint_interval,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        device=device,
    )

    metadata = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "algorithm": "PPO",
        "method": (
            "PPO residual-torque dance tracking with unsaturated tracking rewards, "
            "parallel CPU MuJoCo workers, passive joint damping, pelvis pose stabilization, "
            "randomized gains, and pelvis pushes"
        ),
        "disturbances_enabled": not args.without_disturbances,
        "reference_motion": str(motion_path),
        "robot_model": str(model_xml),
        "final_model": str(model_path.with_suffix(".zip")),
        "checkpoint_dir": str(run_checkpoint_dir),
        "monitor_dir": str(run_monitor_dir),
        "tensorboard_dir": str(TENSORBOARD_DIR),
        "vector_env_type": vector_env_type,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "physics_backend": "MuJoCo CPU",
        "policy_device": device,
        "ppo": asdict(config),
    }
    config_path = CONFIG_DIR / f"training_{run_id}.json"
    config_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Run ID: {run_id}")
    print(f"PPO policy device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"MuJoCo vector environments: {vector_env_type} ({args.n_envs} workers)")
    print("MuJoCo physics runs on the CPU; PPO policy optimization runs on CUDA.")
    print(f"Rollout size: {rollout_size}; batch size: {args.batch_size}")
    print(f"Reference motion: {motion_path}")
    print(f"Experiment metadata: {config_path}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_interval // args.n_envs, 1),
        save_path=str(run_checkpoint_dir),
        name_prefix="g1_dance_ppo",
    )
    callbacks = CallbackList([checkpoint_callback, _build_dance_metrics_callback()])
    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.entropy_coefficient,
            vf_coef=args.vf_coef,
            target_kl=args.target_kl,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs=build_g1_dance_policy_kwargs(
                config.hidden_sizes, config.policy_initial_log_std
            ),
            tensorboard_log=str(TENSORBOARD_DIR),
            seed=args.seed,
            device=device,
            verbose=1,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=f"g1_dance_ppo_{run_id}",
            reset_num_timesteps=True,
        )
        model.save(str(model_path))
        print(f"Final PPO model saved to: {model_path.with_suffix('.zip')}")
        print(f"Checkpoints saved under: {run_checkpoint_dir}")
        print(f"TensorBoard logs saved under: {TENSORBOARD_DIR}")
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def main() -> None:
    """Run PPO motion-tracking training."""
    train(parse_args())


if __name__ == "__main__":
    main()