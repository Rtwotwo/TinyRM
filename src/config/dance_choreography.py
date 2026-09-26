"""Configuration for G1 procedural dance references and motion tracking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DanceSection:
    """Named time interval in the generated dance choreography."""

    name: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class G1DanceChoreographyConfig:
    """Timing and motion limits for a multi-section G1 dance routine."""

    fps: int = 30
    duration_seconds: float = 32.0
    beats_per_minute: float = 112.0
    transition_seconds: float = 0.35
    seed: int = 42
    sections: tuple[DanceSection, ...] = (
        DanceSection("intro_and_arm_raise", 0.0, 4.0),
        DanceSection("step_touch_and_hip_groove", 4.0, 10.0),
        DanceSection("alternating_diagonal_punches", 10.0, 16.0),
        DanceSection("alternating_knee_lifts", 16.0, 22.0),
        DanceSection("overhead_arm_wave", 22.0, 26.0),
        DanceSection("fast_footwork_and_double_punch", 26.0, 30.0),
        DanceSection("wide_final_pose", 30.0, 32.0),
    )
    maximum_joint_fraction: float = 0.78


DEFAULT_DANCE_CHOREOGRAPHY = G1DanceChoreographyConfig()
