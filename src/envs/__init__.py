"""Expose grid and G1 environments without eager optional imports."""

__all__ = ["GridRobotEnv", "G1MotionTrackingEnv"]


def __getattr__(name: str):
    """Load an environment only when callers request its package export."""
    if name == "GridRobotEnv":
        from src.envs.grid_robot import GridRobotEnv

        return GridRobotEnv
    if name == "G1MotionTrackingEnv":
        from src.envs.g1_motion_tracking_env import G1MotionTrackingEnv

        return G1MotionTrackingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
