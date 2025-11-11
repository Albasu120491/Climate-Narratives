"""
Climate Finance Discourse Analysis
Actor-Frame-Argument Pipeline for Longitudinal Analysis
"""

__version__ = "1.0.0"

from .extraction import AFAPipeline, ActorStanceExtractor, FrameClassifier, ArgumentExtractor
from .evaluation import DVFEvaluator, compute_metrics
from .corpus import CorpusSampler, DJIDFilter
from .analysis import TemporalAnalyzer, FrameAnalyzer, ArgumentAnalyzer

__all__ = [
    "AFAPipeline",
    "ActorStanceExtractor",
    "FrameClassifier",
    "ArgumentExtractor",
    "DVFEvaluator",
    "compute_metrics",
    "CorpusSampler",
    "DJIDFilter",
    "TemporalAnalyzer",
    "FrameAnalyzer",
    "ArgumentAnalyzer",
]
