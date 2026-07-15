"""Humanizer - AI text humanization with statistical proof."""

from .features import FeatureExtractor, TextFeatures, feature_vector
from .benchmark import (
    BenchmarkHarness,
    HeuristicDetector,
    PerplexityProxy,
    BenchmarkResult,
    DetectorScore,
    DetectionResult,
)
from .perturb import Humanizer

__version__ = "0.1.0"
__all__ = [
    "FeatureExtractor",
    "TextFeatures",
    "feature_vector",
    "BenchmarkHarness",
    "HeuristicDetector",
    "PerplexityProxy",
    "BenchmarkResult",
    "DetectorScore",
    "DetectionResult",
    "Humanizer",
]