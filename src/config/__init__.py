"""Configuration package for procedural G1 dance training."""

from src.config.dance_choreography import (
    DEFAULT_DANCE_CHOREOGRAPHY,
    DanceSection,
    G1DanceChoreographyConfig,
)
from src.config.spring_dance import (
    DEFAULT_G1_DANCE_PPO_CONFIG,
    G1DancePPOConfig,
)

__all__ = [
    "DEFAULT_DANCE_CHOREOGRAPHY",
    "DEFAULT_G1_DANCE_PPO_CONFIG",
    "DanceSection",
    "G1DanceChoreographyConfig",
    "G1DancePPOConfig",
]