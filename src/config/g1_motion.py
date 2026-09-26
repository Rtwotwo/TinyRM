"""Paths for the retained G1 procedural dance reference generator."""

from src.config.paths import PROJECT_ROOT

EXP_DIR = PROJECT_ROOT / "exp" / "spring_dance"
PROCEDURAL_DANCE_DIR = EXP_DIR / "project" / "procedural_g1_dance"
DEFAULT_TRACK_MOTION = PROCEDURAL_DANCE_DIR / "robot_motion_track_1.pkl"
CONFIG_DIR = EXP_DIR / "configs"
G1_MODEL_XML = PROJECT_ROOT / "agents" / "robots" / "unitree_g1" / "g1_mocap_29dof.xml"
