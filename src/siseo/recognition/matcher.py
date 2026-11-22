"""Nearest-template recognition with rejection.

Every training take is kept as a template (a sequence of band-energy
vectors). A new word is compared against all of them with DTW; each command
scores its closest template. The word is accepted only when

1. it is close enough to that command's own takes (a per-command threshold
   learned from how much the takes differ among themselves), and
2. it is clearly closer to the winner than to the runner-up (confidence).

Otherwise the answer is "not recognized", which is what a game controller
needs: doing nothing is better than turning the wrong way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..dsp.dtw import dtw_distance


@dataclass(frozen=True)
class MatchConfig:
    sharpness: float = 6.0
    """How fast confidence drops as the runner-up gets closer to the winner."""
    min_confidence: float = 0.55
    threshold_slack: float = 1.3
    """Multiplier on the worst within-command distance seen during training."""
    default_threshold: float = 1.3
    """Acceptance distance for commands with a single take (nothing to learn from)."""
    min_takes: int = 1


@dataclass(frozen=True)
class Score:
    command: str
    distance: float
    threshold: float


@dataclass(frozen=True)
class Recognition:
    command: str | None
    """The accepted command, or None."""
    best: str | None
    """The closest command even if rejected (useful to explain rejections)."""
    confidence: float
    reason: str
    """'ok', 'silence', 'unknown' (too far), 'ambiguous' (too close to runner-up) or 'untrained'."""
    scores: list[Score] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.command is not None


def leave_one_out(takes: Sequence[np.ndarray]) -> list[float]:
    """Distance from each take to its nearest sibling take."""
    out = []
    for i, take in enumerate(takes):
        others = [dtw_distance(take, t) for j, t in enumerate(takes) if j != i]
        if others:
            out.append(min(others))
    return out


def learn_threshold(takes: Sequence[np.ndarray], cfg: MatchConfig = MatchConfig()) -> float:
    """Acceptance distance for one command, from the spread of its own takes."""
    spread = leave_one_out(takes)
    if not spread:
        return cfg.default_threshold
    return float(max(spread) * cfg.threshold_slack + 0.15)


class TemplateMatcher:
    def __init__(
        self,
        templates: Mapping[str, Sequence[np.ndarray]],
        thresholds: Mapping[str, float] | None = None,
        cfg: MatchConfig = MatchConfig(),
    ):
        self.cfg = cfg
        self.templates = {k: list(v) for k, v in templates.items() if len(v) >= cfg.min_takes}
        self.thresholds = {
            k: float(thresholds[k]) if thresholds and k in thresholds else learn_threshold(v, cfg)
            for k, v in self.templates.items()
        }

    @property
    def commands(self) -> list[str]:
        return sorted(self.templates)

    def score(self, features: np.ndarray) -> list[Score]:
        """Closest-template distance for every command, best first."""
        scores = [
            Score(cmd, min(dtw_distance(features, t) for t in takes), self.thresholds[cmd])
            for cmd, takes in self.templates.items()
        ]
        return sorted(scores, key=lambda s: s.distance)

    def confidence(self, scores: Sequence[Score]) -> float:
        """Softmax over relative distances: 1.0 when the winner stands alone."""
        if not scores:
            return 0.0
        d = np.array([s.distance for s in scores])
        rel = d / max(d[0], 1e-9) - 1.0
        weights = np.exp(-self.cfg.sharpness * rel)
        return float(weights[0] / weights.sum())

    def recognize(self, features: np.ndarray | None) -> Recognition:
        if not self.templates:
            return Recognition(None, None, 0.0, "untrained")
        if features is None or len(features) == 0:
            return Recognition(None, None, 0.0, "silence")
        scores = self.score(features)
        best = scores[0]
        conf = self.confidence(scores)
        if best.distance > best.threshold:
            return Recognition(None, best.command, conf, "unknown", scores)
        if conf < self.cfg.min_confidence:
            return Recognition(None, best.command, conf, "ambiguous", scores)
        return Recognition(best.command, best.command, conf, "ok", scores)
