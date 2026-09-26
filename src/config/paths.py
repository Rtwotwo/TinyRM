"""Project root paths shared by the G1 dance workflow."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = PROJECT_ROOT / "agents"
ROBOTS_DIR = AGENTS_DIR / "robots"
EXP_DIR = PROJECT_ROOT / "exp"