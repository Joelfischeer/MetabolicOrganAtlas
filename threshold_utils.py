"""
threshold_utils.py
===================
Shared helpers for the MIN_BOOTSTRAP_MEAN threshold used by the condition
config files (config_healthy.py / config_obese.py) across
Edge_general_reference_network and Edge_metabolism_reference_network.

Config files can set a fixed numeric threshold as usual:

    MIN_BOOTSTRAP_MEAN = 100

...or write the bare name `Elbow` to have the threshold picked automatically
from the elbow of that condition's own bootstrap mean distribution:

    MIN_BOOTSTRAP_MEAN = Elbow

`Elbow` does not need to be imported in the config file — `_load_config()` in
each run_network.py / run_comparison.py injects it into the config
module's namespace before the file executes.
"""

import math


class _ElbowSentinel:
    """Marks MIN_BOOTSTRAP_MEAN for automatic elbow-based selection."""

    def __repr__(self) -> str:
        return "Elbow"

    __str__ = __repr__

    # Comparisons against an unresolved sentinel must never raise — before
    # resolve_min_bootstrap_mean() replaces it with a real number (e.g. while
    # a fresh bootstrap run is still in progress), treat it as "nothing
    # qualifies yet" rather than crashing on `mean >= cfg.MIN_BOOTSTRAP_MEAN`.
    def _never(self, other) -> bool:
        return False

    __lt__ = __le__ = __gt__ = __ge__ = __eq__ = _never
    __hash__ = object.__hash__


Elbow = _ElbowSentinel()


def is_elbow(raw_value) -> bool:
    """True if a config's MIN_BOOTSTRAP_MEAN requests elbow auto-selection."""
    return raw_value is Elbow or (
        isinstance(raw_value, str) and raw_value.strip().lower() == "elbow"
    )


def kneedle_elbow(sorted_means: list[float]) -> float:
    """
    Elbow detection on a descending-sorted list of bootstrap means.

    Two adjustments make the result less strict than raw kneedle:
      1. Rolling-mean smoothing (window 5) removes the steep initial cliff
         caused by a handful of very strong pairs, so those no longer
         dominate the distance calculation.
      2. Log-transform of the y-axis compresses the high-value range,
         shifting the point of maximum curvature rightward — capturing
         more edges above the suggested threshold. Uses plain log(v), not
         log1p(v): log1p(v) ≈ v for v ≪ 1 (e.g. Otsuka–Ochiai coefficients,
         typically 0.0001–0.05), which silently disables the compression and
         collapses the elbow to the very first point. Values are already
         filtered to v > 0 below, so log(v) is always defined.
    """
    # Note: no rounding here — callers cover magnitudes from raw counts in
    # the hundreds down to Otsuka–Ochiai coefficients as small as ~0.001.
    # A fixed round(x, 1) would collapse small-magnitude thresholds to 0.0.
    vals = [v for v in sorted_means if v > 0]
    if len(vals) < 3:
        return vals[0] if vals else 0.0
    n = len(vals)

    # Step 1 — smooth
    w = min(5, max(1, n // 6))
    smoothed = [
        sum(vals[max(0, i - w): i + w + 1]) / len(vals[max(0, i - w): i + w + 1])
        for i in range(n)
    ]

    # Step 2 — log transform (plain log, not log1p — see docstring)
    log_vals = [math.log(v) for v in smoothed]
    max_v, min_v = log_vals[0], log_vals[-1]
    if max_v == min_v:
        return vals[0]

    xs = [i / (n - 1) for i in range(n)]
    ys = [(v - min_v) / (max_v - min_v) for v in log_vals]
    # Diagonal from (0,1) to (1,0): signed distance = x + y - 1
    distances = [xs[i] + ys[i] - 1 for i in range(n)]
    elbow_idx = distances.index(max(distances))
    return vals[elbow_idx]


def resolve_min_bootstrap_mean(raw_value, means: list[float]):
    """
    Resolve a config's MIN_BOOTSTRAP_MEAN: pass numeric values through
    unchanged, or compute the elbow-suggested threshold from `means` when the
    config used the `Elbow` sentinel.
    """
    if is_elbow(raw_value):
        sorted_means = sorted((m for m in means if m > 0), reverse=True)
        return round(kneedle_elbow(sorted_means), 6)
    return raw_value
