"""MuJoCo environment for closed-loop G1 dance tracking and recovery."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two scalar-first quaternions."""
    left_w, left_v = float(left[0]), left[1:]
    right_w, right_v = float(right[0]), right[1:]
    scalar = left_w * right_w - float(np.dot(left_v, right_v))
    vector = left_w * right_v + right_w * left_v + np.cross(left_v, right_v)
    return np.concatenate((np.array([scalar]), vector))


def _quaternion_rotation_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the shortest world-frame rotation vector from actual to target."""
    inverse_actual = actual.copy()
    inverse_actual[1:] *= -1.0
    relative = _quaternion_multiply(target, inverse_actual)
    if relative[0] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, max(float(relative[0]), 0.0))
    return relative[1:] * (angle / vector_norm)


class G1MotionTrackingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Track a G1 dance reference with PD control and learned residual torque.

    A low-level pelvis stabilizer follows the reference root pose.
    The policy controls the
    29 actuated joints and learns to maintain choreography under randomized resets,
    controller gains, and external pushes.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        motion_path: str | Path,
        model_path: str | Path,
        *,
        episode_seconds: float = 10.0,
        pd_kp: float = 80.0,
        pd_kd: float = 2.0,
        residual_torque_scale: float = 0.25,
        action_penalty_coefficient: float = 0.03,
        action_smoothness_penalty_coefficient: float = 0.01,
        observation_scales: tuple[float, ...] = (
            0.5, 5.0, 1.5, 5.0, 0.5, 3.0, 5.0
        ),
        reset_joint_noise: float = 0.02,
        root_xy_reset_noise: float = 0.0,
        gain_randomization: float = 0.0,
        joint_passive_damping: float = 3.0,
        root_position_kp: float = 100.0,
        root_position_kd: float = 20.0,
        root_orientation_kp: float = 300.0,
        root_orientation_kd: float = 40.0,
        enable_disturbances: bool = False,
        disturbance_force_min: float = 20.0,
        disturbance_force_max: float = 45.0,
        disturbance_duration_seconds: float = 0.10,
        disturbance_interval_min_seconds: float = 1.5,
        disturbance_interval_max_seconds: float = 3.0,
        fall_height: float = 0.42,
        fall_tilt_radians: float = 1.25,
        random_start: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive.")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode!r}")

        self.motion_path = Path(motion_path).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.motion_path.is_file():
            raise FileNotFoundError(
                f"Reference motion not found: {self.motion_path}. "
                "Run python -m src.trainer.generate_g1_dance_motion, or provide "
                "a GMR motion file."
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")

        with self.motion_path.open("rb") as motion_file:
            motion: dict[str, Any] = pickle.load(motion_file)

        self.reference_root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
        self.reference_root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
        self.reference_joint_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
        self.fps = float(motion.get("fps", 30.0))
        if self.fps <= 0 or len(self.reference_joint_pos) < 2:
            raise ValueError("Motion must contain at least two frames and a positive fps.")
        if self.reference_root_pos.shape != (len(self.reference_joint_pos), 3):
            raise ValueError("root_pos must have shape (frames, 3).")
        if self.reference_root_rot.shape != (len(self.reference_joint_pos), 4):
            raise ValueError("root_rot must have shape (frames, 4) in MuJoCo wxyz order.")
        if not all(
            np.isfinite(array).all()
            for array in (
                self.reference_root_pos,
                self.reference_root_rot,
                self.reference_joint_pos,
            )
        ):
            raise ValueError("Reference motion contains NaN or infinite values.")

        quaternion_norm = np.linalg.norm(self.reference_root_rot, axis=1, keepdims=True)
        if np.any(quaternion_norm < 1.0e-8):
            raise ValueError("Reference motion contains a zero-length root quaternion.")
        self.reference_root_rot /= quaternion_norm

        self.reference_joint_vel = np.gradient(
            self.reference_joint_pos, axis=0
        ) * self.fps
        self.reference_root_vel = np.gradient(
            self.reference_root_pos, axis=0
        ) * self.fps
        self.reference_root_ang_vel = np.zeros_like(self.reference_root_pos)
        for frame_index in range(len(self.reference_root_rot) - 1):
            self.reference_root_ang_vel[frame_index] = (
                _quaternion_rotation_error(
                    self.reference_root_rot[frame_index + 1],
                    self.reference_root_rot[frame_index],
                )
                * self.fps
            )
        self.reference_root_ang_vel[-1] = self.reference_root_ang_vel[-2]

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_passive_damping = float(joint_passive_damping)
        if self.joint_passive_damping < 0.0:
            raise ValueError("joint_passive_damping must be non-negative.")
        # The imported G1 mocap model omits passive joint damping; add a realistic
        # viscous term to prevent low-inertia wrist and ankle joints from ringing.
        self.model.dof_damping[6:] = self.joint_passive_damping
        self.root_position_kp = float(root_position_kp)
        self.root_position_kd = float(root_position_kd)
        self.root_orientation_kp = float(root_orientation_kp)
        self.root_orientation_kd = float(root_orientation_kd)
        if min(
            self.root_position_kp,
            self.root_position_kd,
            self.root_orientation_kp,
            self.root_orientation_kd,
        ) < 0.0:
            raise ValueError("Root stabilizer gains must be non-negative.")
        self.robot_total_mass = float(np.sum(self.model.body_mass))
        self.n_joints = self.reference_joint_pos.shape[1]
        if self.model.nq != 7 + self.n_joints or self.model.nv != 6 + self.n_joints:
            raise ValueError(
                "The G1 MJCF must contain one free root joint followed by the same "
                f"number of scalar joints as the motion ({self.n_joints}); "
                f"model nq/nv are {self.model.nq}/{self.model.nv}."
            )
        if self.model.nu != self.n_joints:
            raise ValueError(
                f"Expected {self.n_joints} G1 actuators; model has {self.model.nu}."
            )

        self.joint_qpos_addresses = np.asarray(self.model.jnt_qposadr[1:], dtype=int)
        self.joint_dof_addresses = np.asarray(self.model.jnt_dofadr[1:], dtype=int)
        joint_force_ranges = np.asarray(self.model.jnt_actfrcrange[1:], dtype=np.float64)
        self.joint_torque_limits = np.max(np.abs(joint_force_ranges), axis=1)
        self.joint_torque_limits = np.where(
            self.joint_torque_limits > 1.0, self.joint_torque_limits, 120.0
        )

        self.frame_skip = max(
            1, int(round(1.0 / (self.fps * float(self.model.opt.timestep))))
        )
        self.episode_frame_limit = min(
            len(self.reference_joint_pos),
            max(1, int(round(episode_seconds * self.fps))),
        )
        self.pd_kp = float(pd_kp)
        self.pd_kd = float(pd_kd)
        self.residual_torque_scale = float(residual_torque_scale)
        self.action_penalty_coefficient = float(action_penalty_coefficient)
        self.action_smoothness_penalty_coefficient = float(action_smoothness_penalty_coefficient)
        self.observation_scales = np.asarray(observation_scales, dtype=np.float64)
        if (
            self.observation_scales.shape != (7,)
            or not np.isfinite(self.observation_scales).all()
            or np.any(self.observation_scales <= 0)
        ):
            raise ValueError("observation_scales must contain seven finite positive values.")
        if min(
            self.action_penalty_coefficient,
            self.action_smoothness_penalty_coefficient,
        ) < 0:
            raise ValueError("Action penalty coefficients must be non-negative.")
        if gain_randomization < 0 or gain_randomization >= 1:
            raise ValueError("gain_randomization must be in [0, 1).")
        if disturbance_force_min < 0 or disturbance_force_max < disturbance_force_min:
            raise ValueError("Disturbance force bounds are invalid.")
        if disturbance_duration_seconds <= 0:
            raise ValueError("disturbance_duration_seconds must be positive.")
        if (
            disturbance_interval_min_seconds <= 0
            or disturbance_interval_max_seconds < disturbance_interval_min_seconds
        ):
            raise ValueError("Disturbance interval bounds are invalid.")
        self.reset_joint_noise = float(reset_joint_noise)
        self.root_xy_reset_noise = float(root_xy_reset_noise)
        self.gain_randomization = float(gain_randomization)
        self.enable_disturbances = bool(enable_disturbances)
        self.disturbance_force_min = float(disturbance_force_min)
        self.disturbance_force_max = float(disturbance_force_max)
        self.disturbance_duration_steps = max(
            1, int(round(disturbance_duration_seconds * self.fps))
        )
        self.disturbance_interval_min_steps = max(
            1, int(round(disturbance_interval_min_seconds * self.fps))
        )
        self.disturbance_interval_max_steps = max(
            self.disturbance_interval_min_steps,
            int(round(disturbance_interval_max_seconds * self.fps)),
        )
        self.active_pd_kp = self.pd_kp
        self.active_pd_kd = self.pd_kd
        self.current_push = np.zeros(3, dtype=np.float64)
        self.push_remaining_steps = 0
        self.next_push_step = 0
        self.fall_height = float(fall_height)
        self.fall_tilt_radians = float(fall_tilt_radians)
        self.random_start = bool(random_start)
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        observation_size = 6 * self.n_joints + 3 + 4 + 6 + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.pelvis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        if self.pelvis_body_id < 0:
            raise ValueError("The G1 model is missing the expected pelvis body.")
        self.renderer: mujoco.Renderer | None = None
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 3.5
        self.camera.azimuth = 135.0
        self.camera.elevation = -12.0

        self.current_frame = 0
        self.start_frame = 0
        self.episode_steps = 0
        self.episode_length = 0
        self.previous_action = np.zeros(self.n_joints, dtype=np.float64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the robot to a reference pose, optionally with small joint noise."""
        super().reset(seed=seed)
        options = options or {}

        latest_start = max(0, len(self.reference_joint_pos) - self.episode_frame_limit)
        if "start_index" in options:
            selected_start = int(options["start_index"])
            if selected_start < 0 or selected_start > latest_start:
                raise ValueError(
                    f"start_index must be between 0 and {latest_start}."
                )
        elif self.random_start and latest_start > 0:
            selected_start = int(self.np_random.integers(0, latest_start + 1))
        else:
            selected_start = 0

        self.start_frame = selected_start
        self.current_frame = selected_start
        gain_scale = self.np_random.uniform(
            1.0 - self.gain_randomization,
            1.0 + self.gain_randomization,
        )
        self.active_pd_kp = self.pd_kp * gain_scale
        self.active_pd_kd = self.pd_kd * gain_scale
        self.current_push.fill(0.0)
        self.push_remaining_steps = 0
        self.next_push_step = int(
            self.np_random.integers(
                self.disturbance_interval_min_steps,
                self.disturbance_interval_max_steps + 1,
            )
        )
        self.episode_steps = 0
        self.episode_length = min(
            self.episode_frame_limit,
            len(self.reference_joint_pos) - selected_start,
        )
        self.previous_action.fill(0.0)

        mujoco.mj_resetData(self.model, self.data)
        frame = self.current_frame
        self.data.qpos[:3] = self.reference_root_pos[frame]
        if self.root_xy_reset_noise > 0:
            self.data.qpos[:2] += self.np_random.uniform(
                -self.root_xy_reset_noise,
                self.root_xy_reset_noise,
                size=2,
            )
        self.data.qpos[3:7] = self.reference_root_rot[frame]
        self.data.qpos[7:] = self.reference_joint_pos[frame]
        if self.reset_joint_noise > 0:
            self.data.qpos[7:] += self.np_random.normal(
                0.0, self.reset_joint_noise, size=self.n_joints
            )
        self.data.qvel[:3] = self.reference_root_vel[frame]
        # Free-joint angular velocity is stored in the local body frame.
        qw, qx, qy, qz = self.reference_root_rot[frame]
        root_rotation = np.array(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qw * qz),
                    2.0 * (qx * qz + qw * qy),
                ],
                [
                    2.0 * (qx * qy + qw * qz),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qw * qx),
                ],
                [
                    2.0 * (qx * qz - qw * qy),
                    2.0 * (qy * qz + qw * qx),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )
        self.data.qvel[3:6] = (
            root_rotation.T @ self.reference_root_ang_vel[frame]
        )
        self.data.qvel[6:] = self.reference_joint_vel[frame]
        mujoco.mj_forward(self.model, self.data)
        return self._get_observation(), {
            "start_frame": int(self.start_frame),
            "fps": float(self.fps),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float | int | bool]]:
        """Apply a learned residual torque and advance one reference frame."""
        clipped_action = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0
        )
        if clipped_action.shape != (self.n_joints,):
            raise ValueError(
                f"Expected action shape {(self.n_joints,)}, got {clipped_action.shape}."
            )

        frame = self.current_frame
        reference_position = self.reference_joint_pos[frame]
        reference_velocity = self.reference_joint_vel[frame]
        residual_torque = (
            clipped_action * self.joint_torque_limits * self.residual_torque_scale
        )

        if (
            self.enable_disturbances
            and self.push_remaining_steps == 0
            and self.episode_steps >= self.next_push_step
        ):
            direction = float(self.np_random.uniform(-np.pi, np.pi))
            magnitude = float(
                self.np_random.uniform(
                    self.disturbance_force_min,
                    self.disturbance_force_max,
                )
            )
            self.current_push[:] = (
                magnitude * np.cos(direction),
                magnitude * np.sin(direction),
                0.0,
            )
            self.push_remaining_steps = self.disturbance_duration_steps

        disturbance_active = self.enable_disturbances and self.push_remaining_steps > 0
        for _ in range(self.frame_skip):
            actual_position = self.data.qpos[self.joint_qpos_addresses]
            actual_velocity = self.data.qvel[self.joint_dof_addresses]
            pd_torque = (
                self.active_pd_kp * (reference_position - actual_position)
                + self.active_pd_kd * (reference_velocity - actual_velocity)
            )
            total_torque = np.clip(
                pd_torque + residual_torque,
                -self.joint_torque_limits,
                self.joint_torque_limits,
            )
            self.data.qfrc_applied.fill(0.0)
            self.data.qfrc_applied[self.joint_dof_addresses] = total_torque

            # A low-level pelvis stabilizer lets the policy focus on dance tracking and recovery.
            root_position_error = self.reference_root_pos[frame] - self.data.qpos[:3]
            root_linear_velocity_error = (
                self.reference_root_vel[frame] - self.data.qvel[:3]
            )
            root_force = self.robot_total_mass * (
                self.root_position_kp * root_position_error
                + self.root_position_kd * root_linear_velocity_error
                - self.model.opt.gravity
            )
            root_rotation_error = _quaternion_rotation_error(
                self.reference_root_rot[frame], self.data.qpos[3:7]
            )
            pelvis_rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
            actual_root_ang_vel_world = pelvis_rotation @ self.data.qvel[3:6]
            root_angular_velocity_error = (
                self.reference_root_ang_vel[frame] - actual_root_ang_vel_world
            )
            root_torque = (
                self.root_orientation_kp * root_rotation_error
                + self.root_orientation_kd * root_angular_velocity_error
            )
            if disturbance_active:
                root_force = root_force + self.current_push
            mujoco.mj_applyFT(
                self.model,
                self.data,
                root_force,
                root_torque,
                self.data.xipos[self.pelvis_body_id],
                self.pelvis_body_id,
                self.data.qfrc_applied,
            )
            mujoco.mj_step(self.model, self.data)

        if disturbance_active:
            self.push_remaining_steps -= 1
            if self.push_remaining_steps == 0:
                interval = int(
                    self.np_random.integers(
                        self.disturbance_interval_min_steps,
                        self.disturbance_interval_max_steps + 1,
                    )
                )
                self.next_push_step = self.episode_steps + interval

        tracking = self._tracking_metrics(frame)
        action_rms = float(np.sqrt(np.mean(np.square(clipped_action))))
        reward_components = {
            "joint_pose_reward": 1.20 * tracking["joint_pose_score"],
            "joint_velocity_reward": 0.20 * tracking["joint_velocity_score"],
            "root_position_reward": 0.25 * tracking["root_position_score"],
            "root_orientation_reward": 0.15 * tracking["root_orientation_score"],
            "action_penalty": -self.action_penalty_coefficient * float(np.mean(np.square(clipped_action))),
            "action_smoothness_penalty": -self.action_smoothness_penalty_coefficient
            * float(np.mean(np.square(clipped_action - self.previous_action))),
        }
        reward = 0.10 + sum(reward_components.values())
        self.previous_action = clipped_action
        self.episode_steps += 1
        self.current_frame += 1

        pelvis_rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        upright_cosine = float(np.clip(pelvis_rotation[2, 2], -1.0, 1.0))
        tilt_angle = float(np.arccos(upright_cosine))
        fell = bool(
            self.data.qpos[2] < self.fall_height
            or tilt_angle > self.fall_tilt_radians
            or not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        )
        if fell:
            reward -= 3.0
        terminated = fell
        truncated = bool(
            not terminated
            and (
                self.episode_steps >= self.episode_length
                or self.current_frame >= len(self.reference_joint_pos)
            )
        )

        info: dict[str, float | int | bool] = {
            "frame": int(frame),
            "joint_rmse": tracking["joint_rmse"],
            "root_position_error": tracking["root_position_error"],
            "root_orientation_error": tracking["root_orientation_error"],
            "joint_pose_score": tracking["joint_pose_score"],
            "joint_velocity_score": tracking["joint_velocity_score"],
            "root_position_score": tracking["root_position_score"],
            "root_orientation_score": tracking["root_orientation_score"],
            "action_rms": action_rms,
            "reward_joint_pose": reward_components["joint_pose_reward"],
            "reward_joint_velocity": reward_components["joint_velocity_reward"],
            "reward_root_position": reward_components["root_position_reward"],
            "reward_root_orientation": reward_components["root_orientation_reward"],
            "disturbance_active": bool(disturbance_active),
            "fell": fell,
        }
        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render one RGB frame for evaluation video output."""
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=960, height=720)
        self.camera.lookat[:] = self.data.qpos[:3] + np.array([0.0, 0.0, 0.55])
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self) -> None:
        """Release the optional MuJoCo off-screen renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _get_observation(self) -> np.ndarray:
        """Build a state/reference observation for the policy."""
        frame = min(self.current_frame, len(self.reference_joint_pos) - 1)
        future_frame = min(
            frame + max(1, int(round(0.15 * self.fps))),
            len(self.reference_joint_pos) - 1,
        )

        current_joint_position = self.data.qpos[7:]
        current_joint_velocity = self.data.qvel[6:]
        reference_position = self.reference_joint_pos[frame]
        reference_velocity = self.reference_joint_vel[frame]
        future_position = self.reference_joint_pos[future_frame]

        actual_quaternion = self.data.qpos[3:7].copy()
        reference_quaternion = self.reference_root_rot[frame]
        if float(np.dot(actual_quaternion, reference_quaternion)) < 0:
            actual_quaternion *= -1.0

        phase = 2.0 * np.pi * frame / max(1, len(self.reference_joint_pos) - 1)
        (
            joint_position_scale,
            joint_velocity_scale,
            reference_position_scale,
            reference_velocity_scale,
            root_position_scale,
            root_linear_velocity_scale,
            root_angular_velocity_scale,
        ) = self.observation_scales
        root_velocity = self.data.qvel[:6].copy()
        root_velocity[:3] /= root_linear_velocity_scale
        root_velocity[3:6] /= root_angular_velocity_scale

        # Scale physical feature groups to comparable ranges before the MLP policy.
        observation = np.concatenate(
            [
                (current_joint_position - reference_position) / joint_position_scale,
                (current_joint_velocity - reference_velocity) / joint_velocity_scale,
                reference_position / reference_position_scale,
                future_position / reference_position_scale,
                reference_velocity / reference_velocity_scale,
                (self.reference_root_pos[frame] - self.data.qpos[:3]) / root_position_scale,
                reference_quaternion - actual_quaternion,
                root_velocity,
                self.previous_action,
                np.array([np.sin(phase), np.cos(phase)]),
            ]
        )
        return np.nan_to_num(
            np.clip(observation, -10.0, 10.0).astype(np.float32),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
    def _tracking_metrics(self, frame: int) -> dict[str, float]:
        """Measure joint and root pose tracking for reward and evaluation logs."""
        joint_position_error = self.data.qpos[7:] - self.reference_joint_pos[frame]
        joint_velocity_error = self.data.qvel[6:] - self.reference_joint_vel[frame]
        root_position_error = self.data.qpos[:3] - self.reference_root_pos[frame]

        actual_quaternion = self.data.qpos[3:7]
        reference_quaternion = self.reference_root_rot[frame]
        quaternion_dot = abs(float(np.dot(actual_quaternion, reference_quaternion)))
        root_orientation_error = float(
            2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
        )
        joint_rmse = float(np.sqrt(np.mean(np.square(joint_position_error))))

        return {
            "joint_rmse": joint_rmse,
            "root_position_error": float(np.linalg.norm(root_position_error)),
            "root_orientation_error": root_orientation_error,
            "joint_pose_score": float(
                np.exp(-np.mean(np.square(joint_position_error)) / 0.01)
            ),
            "joint_velocity_score": float(
                np.exp(-np.mean(np.square(joint_velocity_error)) / 1.0)
            ),
            "root_position_score": float(
                np.exp(-np.mean(np.square(root_position_error)) / 0.0025)
            ),
            "root_orientation_score": float(
                np.exp(-np.square(root_orientation_error) / 0.04)
            ),
        }

