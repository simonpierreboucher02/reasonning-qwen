"""
Inference module for text generation and sampling.

This module provides utilities for generating text from the model using
various sampling strategies.
"""

from .generator import TextGenerator
from .sampler import (
    SamplingStrategy,
    GreedySampler,
    TemperatureSampler,
    TopPSampler,
    SelfConsistencySampler,
)

__all__ = [
    "TextGenerator",
    "SamplingStrategy",
    "GreedySampler",
    "TemperatureSampler",
    "TopPSampler",
    "SelfConsistencySampler",
]
