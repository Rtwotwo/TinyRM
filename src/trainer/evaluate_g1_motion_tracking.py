"""Evaluate a trained PPO dance-tracking policy and optionally record a video."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

import imageio.v2 as imageio
import torch
from stable_baselines3 import PPO

from src.config.spring_dance import (
    DEFAULT_G1_DANCE_PPO_CONFIG,
    DEFAULT_TRACK_MOTION,
    EVALUATION_DIR,
    G1_MODEL_XML,
    MODEL_DIR,
    VIDEO_DIR,
)
from src.envs.g1_motion_tracking_env import G1MotionTrackingEnv


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO controller against one G1 reference motion."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--motion", type=Path, default=DEFAULT_TRACK_MOTION)
    parser.add_argument("--model-xml", type=Path, default=G1_MODEL_XML)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-video", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Path:
    """Run deterministic rollouts and store a JSON report under exp/."""
    model_path = args.model.expanduser().resolve()
    motion_path = args.motion.expanduser().resolve()
    model_xml = args.model_xml.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    if not motion_path.is_file():
        raise FileNotFoundError(f"Reference motion not found: {motion_path}")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested, but no CUDA device is available.")

    env = G1MotionTrackingEnv(
        motion_path=motion_path,
        model_path=model_xml,
        episode_seconds=1.0e6,
        random_start=False,
        joint_passive_damping=DEFAULT_G1_DANCE_PPO_CONFIG.joint_passive_damping,
        action_penalty_coefficient=DEFAULT_G1_DANCE_PPO_CONFIG.action_penalty_coefficient,
        action_smoothness_penalty_coefficient=DEFAULT_G1_DANCE_PPO_CONFIG.action_smoothness_penalty_coefficient,
        observation_scales=DEFAULT_G1_DANCE_PPO_CONFIG.observation_scales,
        root_position_kp=DEFAULT_G1_DANCE_PPO_CONFIG.root_position_kp,
        root_position_kd=DEFAULT_G1_DANCE_PPO_CONFIG.root_position_kd,
        root_orientation_kp=DEFAULT_G1_DANCE_PPO_CONFIG.root_orientation_kp,
        root_orientation_kd=DEFAULT_G1_DANCE_PPO_CONFIG.root_orientation_kd,
        render_mode="rgb_array" if args.render_video else None,
    )
    model = PPO.load(str(model_path), env=None, device=args.device)

    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_DIR / f"g1_dance_policy_{run_id}.mp4" if args.render_video else None

    episode_reports: list[dict[str, float | int | bool]] = []
    try:
        for episode_index in range(args.episodes):
            observation, _ = env.reset(seed=episode_index, options={"start_index": 0})
            total_return = 0.0
            steps = 0
            joint_errors: list[float] = []
            root_errors: list[float] = []
            orientation_errors: list[float] = []
            fell = False

            writer = (
                imageio.get_writer(
                    str(video_path),
                    fps=env.fps,
                    codec="libx264",
                    quality=8,
                )
                if args.render_video and episode_index == 0
                else None
            )
            try:
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_return += float(reward)
                    steps += 1
                    joint_errors.append(float(info["joint_rmse"]))
                    root_errors.append(float(info["root_position_error"]))
                    orientation_errors.append(float(info["root_orientation_error"]))
                    fell = fell or bool(info["fell"])

                    if writer is not None:
                        frame = env.render()
                        if frame is not None:
                            writer.append_data(frame)
                    if terminated or truncated:
                        break
            finally:
                if writer is not None:
                    writer.close()

            episode_reports.append(
                {
                    "episode": episode_index + 1,
                    "return": total_return,
                    "steps": steps,
                    "mean_joint_rmse_rad": mean(joint_errors) if joint_errors else 0.0,
                    "mean_root_position_error_m": mean(root_errors) if root_errors else 0.0,
                    "mean_root_orientation_error_rad": (
                        mean(orientation_errors) if orientation_errors else 0.0
                    ),
                    "fell": fell,
                }
            )
    finally:
        env.close()

    report = {
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model_path),
        "reference_motion": str(motion_path),
        "robot_model": str(model_xml),
        "policy_device": args.device,
        "episodes": episode_reports,
        "mean_return": mean(float(item["return"]) for item in episode_reports),
        "mean_joint_rmse_rad": mean(
            float(item["mean_joint_rmse_rad"]) for item in episode_reports
        ),
        "fall_rate": mean(float(bool(item["fell"])) for item in episode_reports),
        "video": str(video_path) if video_path is not None else None,
    }
    report_path = EVALUATION_DIR / f"g1_dance_evaluation_{run_id}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Evaluation report saved to: {report_path}")
    if video_path is not None:
        print(f"Evaluation video saved to: {video_path}")
    return report_path


def main() -> None:
    """Run deterministic PPO evaluation."""
    evaluate(parse_args())


if __name__ == "__main__":
    main()

