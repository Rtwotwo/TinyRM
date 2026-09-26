"""Configuration for procedural G1 dance references and PPO tracking."""

from dataclasses import dataclass

from src.config.paths import PROJECT_ROOT

EXP_DIR = PROJECT_ROOT / "exp" / "spring_dance"
PROCEDURAL_DANCE_DIR = EXP_DIR / "project" / "procedural_g1_dance"
PROCEDURAL_DANCE_MOTION = PROCEDURAL_DANCE_DIR / "robot_motion_track_1.pkl"
VIDEO_DIR = EXP_DIR / "videos"
MODEL_DIR = EXP_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
TENSORBOARD_DIR = EXP_DIR / "logs" / "tensorboard"
MONITOR_DIR = EXP_DIR / "logs" / "monitor"
CONFIG_DIR = EXP_DIR / "configs"
EVALUATION_DIR = EXP_DIR / "evaluations"
G1_MODEL_XML = PROJECT_ROOT / "agents" / "robots" / "unitree_g1" / "g1_mocap_29dof.xml"
DEFAULT_TRACK_MOTION = PROCEDURAL_DANCE_MOTION


@dataclass(frozen=True)
class G1DancePPOConfig:
    """Hyperparameters for PPO tracking of a generated G1 dance."""

    total_timesteps: int = 2_000_000
    n_envs: int = 6
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 10
    learning_rate: float = 2.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    entropy_coefficient: float = 0.0005
    target_kl: float = 0.015
    value_function_coefficient: float = 0.75
    checkpoint_interval: int = 100_000
    episode_seconds: float = 10.0
    hidden_sizes: tuple[int, ...] = (512, 512, 256)
    policy_initial_log_std: float = -0.7
    action_penalty_coefficient: float = 0.03
    action_smoothness_penalty_coefficient: float = 0.01
    # Scales: joint position error, joint velocity error, reference position,
    # reference velocity, root position error, root linear velocity, root angular velocity.
    observation_scales: tuple[float, ...] = (0.5, 5.0, 1.5, 5.0, 0.5, 3.0, 5.0)
    seed: int = 42
    device: str = "cuda"
    pd_kp: float = 65.0
    pd_kd: float = 1.8
    residual_torque_scale: float = 0.20
    joint_passive_damping: float = 3.0
    root_position_kp: float = 100.0
    root_position_kd: float = 20.0
    root_orientation_kp: float = 300.0
    root_orientation_kd: float = 40.0
    reset_joint_noise: float = 0.05
    root_xy_reset_noise: float = 0.025
    gain_randomization: float = 0.15
    disturbance_force_min: float = 20.0
    disturbance_force_max: float = 45.0
    disturbance_duration_seconds: float = 0.10
    disturbance_interval_min_seconds: float = 1.5
    disturbance_interval_max_seconds: float = 3.0
    fall_height: float = 0.42
    fall_tilt_radians: float = 1.25


DEFAULT_G1_DANCE_PPO_CONFIG = G1DancePPOConfig()