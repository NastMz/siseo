"""Recognition: analysis of a clip and nearest-template matching with rejection."""

from .analysis import Analysis, Analyzer
from .matcher import MatchConfig, Recognition, Score, TemplateMatcher, learn_threshold

__all__ = [
    "Analysis",
    "Analyzer",
    "MatchConfig",
    "Recognition",
    "Score",
    "TemplateMatcher",
    "learn_threshold",
]
