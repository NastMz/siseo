"""Dynamic time warping between two feature sequences.

Nobody says a word at the same speed twice. DTW finds the alignment of two
band-energy sequences (stretching and compressing time) that minimizes the
summed frame-to-frame distance, so a slow "arriba" still matches a fast one.
"""

from __future__ import annotations

import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Length-normalized DTW cost between sequences of shape ``(n, d)`` and ``(m, d)``.

    Steps are (1,0), (0,1) and (1,1); the local cost is the Euclidean distance
    between frames. The total is divided by ``n + m`` so short and long words
    are comparable. Each row is solved in vector form: the horizontal
    recurrence ``D[i,j] = min(t[j], c[i,j] + D[i,j-1])`` equals
    ``S[j] + min_{k<=j}(t[k] - S[k])`` with ``S`` the running sum of ``c[i]``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("sequences must be 2-D with the same feature dimension")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    cost = np.sqrt(np.clip(sq, 0.0, None))

    prev = np.cumsum(cost[0])
    for i in range(1, n):
        diag = np.concatenate(([np.inf], prev[:-1]))
        t = cost[i] + np.minimum(prev, diag)
        s = np.cumsum(cost[i])
        prev = s + np.minimum.accumulate(t - s)
    return float(prev[-1] / (n + m))
