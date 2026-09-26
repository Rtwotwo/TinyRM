"""Generate a multi-section rhythmic G1 dance reference without external weights."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from src.config.dance_choreography import DEFAULT_DANCE_CHOREOGRAPHY
from src.config.g1_motion import (
    CONFIG_DIR,
    DEFAULT_TRACK_MOTION,
    G1_MODEL_XML,
)


def _smoothstep(value: float) -> float:
    """Return a smooth cubic interpolation value between zero and one."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _section_weight(
    time_seconds: float,
    start: float,
    end: float,
    fade_seconds: float,
    *,
    hold_at_end: bool = False,
) -> float:
    """Build a smooth envelope so adjacent dance phrases do not jump."""
    if time_seconds < start or time_seconds > end:
        return 0.0
    fade_in = _smoothstep((time_seconds - start) / max(fade_seconds, 1.0e-6))
    if hold_at_end:
        return fade_in
    fade_out = _smoothstep((end - time_seconds) / max(fade_seconds, 1.0e-6))
    return min(fade_in, fade_out)


def _phrase_offsets(
    section_index: int,
    local_time: float,
    *,
    bpm: float,
) -> dict[str, float]:
    """Return joint-space accents for one phrase of the choreography."""
    beat_frequency = bpm / 60.0
    beat = 2.0 * np.pi * beat_frequency * local_time
    half_beat = 2.0 * np.pi * (beat_frequency / 2.0) * local_time
    alternating = float(np.sin(beat))
    left_active = max(0.0, alternating)
    right_active = max(0.0, -alternating)

    if section_index == 0:
        raise RuntimeError("The intro pose is computed from phrase progress.")
    if section_index == 1:
        sway = float(np.sin(half_beat))
        return {
            "left_hip_roll_joint": 0.12 * sway,
            "right_hip_roll_joint": -0.12 * sway,
            "left_hip_yaw_joint": 0.08 * sway,
            "right_hip_yaw_joint": -0.08 * sway,
            "left_knee_joint": 0.12 + 0.12 * left_active,
            "right_knee_joint": 0.12 + 0.12 * right_active,
            "left_ankle_pitch_joint": -0.08 * left_active,
            "right_ankle_pitch_joint": -0.08 * right_active,
            "waist_yaw_joint": 0.17 * sway,
            "waist_roll_joint": 0.08 * sway,
            "left_shoulder_pitch_joint": -0.38 - 0.18 * sway,
            "right_shoulder_pitch_joint": -0.38 + 0.18 * sway,
            "left_shoulder_roll_joint": 0.18 + 0.08 * sway,
            "right_shoulder_roll_joint": -0.18 + 0.08 * sway,
            "left_elbow_joint": 0.62 + 0.18 * right_active,
            "right_elbow_joint": 0.62 + 0.18 * left_active,
        }
    if section_index == 2:
        return {
            "left_hip_yaw_joint": -0.08 * left_active + 0.08 * right_active,
            "right_hip_yaw_joint": 0.08 * left_active - 0.08 * right_active,
            "left_knee_joint": 0.08,
            "right_knee_joint": 0.08,
            "waist_yaw_joint": 0.22 * (right_active - left_active),
            "waist_roll_joint": 0.05 * alternating,
            "left_shoulder_pitch_joint": -0.55 - 0.32 * left_active,
            "right_shoulder_pitch_joint": -0.55 - 0.32 * right_active,
            "left_shoulder_roll_joint": 0.28 + 0.12 * left_active,
            "right_shoulder_roll_joint": -0.28 - 0.12 * right_active,
            "left_shoulder_yaw_joint": -0.18 * left_active,
            "right_shoulder_yaw_joint": 0.18 * right_active,
            "left_elbow_joint": 0.95 - 0.70 * left_active,
            "right_elbow_joint": 0.95 - 0.70 * right_active,
            "left_wrist_roll_joint": 0.15 * left_active,
            "right_wrist_roll_joint": -0.15 * right_active,
        }
    if section_index == 3:
        lift_wave = float(np.sin(half_beat))
        lift_left = max(0.0, lift_wave)
        lift_right = max(0.0, -lift_wave)
        return {
            "left_hip_pitch_joint": -0.18 * lift_left + 0.06 * lift_right,
            "right_hip_pitch_joint": -0.18 * lift_right + 0.06 * lift_left,
            "left_hip_roll_joint": 0.08 * lift_left,
            "right_hip_roll_joint": -0.08 * lift_right,
            "left_knee_joint": 0.10 + 0.48 * lift_left,
            "right_knee_joint": 0.10 + 0.48 * lift_right,
            "left_ankle_pitch_joint": -0.10 * lift_left,
            "right_ankle_pitch_joint": -0.10 * lift_right,
            "waist_yaw_joint": 0.13 * np.sin(half_beat + np.pi / 2),
            "left_shoulder_pitch_joint": -0.68 - 0.22 * lift_right,
            "right_shoulder_pitch_joint": -0.68 - 0.22 * lift_left,
            "left_shoulder_roll_joint": 0.32 + 0.10 * lift_right,
            "right_shoulder_roll_joint": -0.32 - 0.10 * lift_left,
            "left_elbow_joint": 0.72 + 0.28 * lift_right,
            "right_elbow_joint": 0.72 + 0.28 * lift_left,
        }
    if section_index == 4:
        wave = float(np.sin(half_beat))
        left_wave = max(0.0, wave)
        right_wave = max(0.0, -wave)
        return {
            "left_knee_joint": 0.10 + 0.08 * left_wave,
            "right_knee_joint": 0.10 + 0.08 * right_wave,
            "waist_roll_joint": 0.10 * wave,
            "waist_pitch_joint": 0.10 * np.sin(half_beat + np.pi / 2),
            "left_shoulder_pitch_joint": -0.95 + 0.18 * wave,
            "right_shoulder_pitch_joint": -0.95 - 0.18 * wave,
            "left_shoulder_roll_joint": 0.42 + 0.18 * left_wave,
            "right_shoulder_roll_joint": -0.42 - 0.18 * right_wave,
            "left_shoulder_yaw_joint": -0.24 * left_wave,
            "right_shoulder_yaw_joint": 0.24 * right_wave,
            "left_elbow_joint": 0.42 + 0.48 * right_wave,
            "right_elbow_joint": 0.42 + 0.48 * left_wave,
            "left_wrist_roll_joint": 0.22 * np.sin(beat),
            "right_wrist_roll_joint": -0.22 * np.sin(beat),
        }
    if section_index == 5:
        stomp = float(np.sin(beat))
        stomp_left = max(0.0, stomp)
        stomp_right = max(0.0, -stomp)
        return {
            "left_hip_pitch_joint": -0.12 * stomp_left,
            "right_hip_pitch_joint": -0.12 * stomp_right,
            "left_hip_roll_joint": 0.10 * stomp,
            "right_hip_roll_joint": -0.10 * stomp,
            "left_knee_joint": 0.10 + 0.24 * stomp_left,
            "right_knee_joint": 0.10 + 0.24 * stomp_right,
            "waist_yaw_joint": 0.16 * np.sin(beat / 2),
            "waist_pitch_joint": 0.08 * np.sin(beat),
            "left_shoulder_pitch_joint": -0.52 - 0.28 * stomp_left,
            "right_shoulder_pitch_joint": -0.52 - 0.28 * stomp_right,
            "left_shoulder_roll_joint": 0.26 + 0.12 * stomp_left,
            "right_shoulder_roll_joint": -0.26 - 0.12 * stomp_right,
            "left_elbow_joint": 0.80 - 0.50 * stomp_left,
            "right_elbow_joint": 0.80 - 0.50 * stomp_right,
        }
    if section_index == 6:
        return {
            "left_hip_roll_joint": 0.06,
            "right_hip_roll_joint": -0.06,
            "left_knee_joint": 0.16,
            "right_knee_joint": 0.16,
            "waist_yaw_joint": 0.10,
            "left_shoulder_pitch_joint": -0.88,
            "right_shoulder_pitch_joint": -0.88,
            "left_shoulder_roll_joint": 0.48,
            "right_shoulder_roll_joint": -0.48,
            "left_shoulder_yaw_joint": -0.20,
            "right_shoulder_yaw_joint": 0.20,
            "left_elbow_joint": 0.25,
            "right_elbow_joint": 0.25,
        }
    raise ValueError(f"Unknown dance section index: {section_index}")


def build_dance_motion(
    model_path: Path,
    *,
    duration_seconds: float,
    fps: int,
    bpm: float,
    transition_seconds: float,
    seed: int,
) -> tuple[dict[str, object], list[str]]:
    """Build a smooth GMR-compatible G1 joint trajectory."""
    if duration_seconds <= 0 or fps <= 0 or bpm <= 0:
        raise ValueError("Duration, frame rate, and tempo must be positive.")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    if model.nu != 29 or model.nq != 36 or model.nv != 35:
        raise ValueError(
            f"Expected the 29-DoF G1 model, got nq={model.nq}, nv={model.nv}, nu={model.nu}."
        )

    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, model.njnt)
    ]
    joint_index = {name: index for index, name in enumerate(joint_names)}
    neutral = data.qpos[7:].copy()
    joint_limits = np.asarray(model.jnt_range[1:], dtype=np.float64)
    if len(joint_names) != 29:
        raise ValueError("The G1 scene must expose 29 scalar joints after the free root.")

    frame_count = int(round(duration_seconds * fps)) + 1
    times = np.arange(frame_count, dtype=np.float64) / fps
    target_positions = np.repeat(neutral[None, :], frame_count, axis=0)
    phase = 2.0 * np.pi * (bpm / 60.0) * times
    global_bounce = 0.025 * (0.5 - 0.5 * np.cos(phase))
    for name in ("left_knee_joint", "right_knee_joint"):
        target_positions[:, joint_index[name]] += 0.05 + global_bounce

    sections = DEFAULT_DANCE_CHOREOGRAPHY.sections
    for frame_index, time_seconds in enumerate(times):
        for section_index, section in enumerate(sections):
            hold_final_pose = section_index == len(sections) - 1
            weight = _section_weight(
                float(time_seconds),
                section.start_seconds,
                min(section.end_seconds, duration_seconds),
                transition_seconds,
                hold_at_end=hold_final_pose,
            )
            if weight <= 0:
                continue
            local_time = max(0.0, float(time_seconds) - section.start_seconds)
            if section_index == 0:
                progress = _smoothstep(
                    local_time / max(section.end_seconds - section.start_seconds, 1.0e-6)
                )
                offsets = {
                    "left_shoulder_pitch_joint": -0.72 * progress,
                    "right_shoulder_pitch_joint": -0.72 * progress,
                    "left_shoulder_roll_joint": 0.28 * progress,
                    "right_shoulder_roll_joint": -0.28 * progress,
                    "left_elbow_joint": 0.58 * progress,
                    "right_elbow_joint": 0.58 * progress,
                    "left_knee_joint": 0.08 * progress,
                    "right_knee_joint": 0.08 * progress,
                }
            else:
                offsets = _phrase_offsets(section_index, local_time, bpm=bpm)
            for joint_name, offset in offsets.items():
                target_positions[frame_index, joint_index[joint_name]] += (
                    weight * float(offset)
                )

    margin = 0.03
    safe_low = joint_limits[:, 0] + margin
    safe_high = joint_limits[:, 1] - margin
    target_positions = np.clip(target_positions, safe_low, safe_high)
    joint_velocities = np.gradient(target_positions, axis=0) * fps
    root_position = np.repeat(data.qpos[:3][None, :], frame_count, axis=0)
    root_rotation = np.zeros((frame_count, 4), dtype=np.float64)
    root_rotation[:, 0] = 1.0

    motion: dict[str, object] = {
        "fps": float(fps),
        "robot_type": "unitree_g1",
        "num_frames": int(frame_count),
        "human_height": 1.8,
        "root_pos": root_position,
        "root_rot": root_rotation,
        "dof_pos": target_positions,
        "joint_names": joint_names,
        "joint_vel": joint_velocities,
        "local_body_pos": np.empty((frame_count, 0, 3), dtype=np.float32),
        "link_body_list": [],
        "motion_source": "procedural_multi_section_dance",
        "dance_sections": [
            {
                "name": section.name,
                "start_seconds": section.start_seconds,
                "end_seconds": min(section.end_seconds, duration_seconds),
            }
            for section in sections
            if section.start_seconds < duration_seconds
        ],
        "seed": int(seed),
    }
    return motion, joint_names


def parse_args() -> argparse.Namespace:
    """Parse reference-motion generation options."""
    defaults = DEFAULT_DANCE_CHOREOGRAPHY
    parser = argparse.ArgumentParser(
        description="Generate a rhythmic multi-section G1 dance reference."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_TRACK_MOTION)
    parser.add_argument("--model-xml", type=Path, default=G1_MODEL_XML)
    parser.add_argument("--duration-seconds", type=float, default=defaults.duration_seconds)
    parser.add_argument("--fps", type=int, default=defaults.fps)
    parser.add_argument("--bpm", type=float, default=defaults.beats_per_minute)
    parser.add_argument("--transition-seconds", type=float, default=defaults.transition_seconds)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate and save the choreography under exp/."""
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    model_path = args.model_xml.expanduser().resolve()
    if output_path.exists() and not args.force:
        print(f"Keeping existing dance reference: {output_path}")
        print("Pass --force to regenerate it.")
        return
    if not model_path.is_file():
        raise FileNotFoundError(f"G1 MuJoCo model not found: {model_path}")

    motion, joint_names = build_dance_motion(
        model_path,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        bpm=args.bpm,
        transition_seconds=args.transition_seconds,
        seed=args.seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as motion_file:
        pickle.dump(motion, motion_file, protocol=4)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    report_path = CONFIG_DIR / f"dance_reference_{run_id}.json"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "motion_path": str(output_path),
        "model_path": str(model_path),
        "duration_seconds": args.duration_seconds,
        "fps": args.fps,
        "bpm": args.bpm,
        "frames": int(motion["num_frames"]),
        "joint_names": joint_names,
        "sections": motion["dance_sections"],
        "seed": args.seed,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Dance reference saved to: {output_path}")
    print(f"Reference metadata saved to: {report_path}")
    print(f"Choreography: {len(motion['dance_sections'])} sections, {args.duration_seconds:.1f}s")
    print(f"Joint trajectory shape: {motion['dof_pos'].shape}")


if __name__ == "__main__":
    main()
