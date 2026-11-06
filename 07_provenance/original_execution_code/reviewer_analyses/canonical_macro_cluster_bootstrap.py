"""Canonical image-level macro estimand with within-class component bootstrap.

The resampling unit is a visual-similarity component within each true class.
For each replicate, sampled component image deltas are pooled within a class,
so a component contributes in proportion to its number of images. Class means
are then averaged with equal class weight. Components are not specimen IDs.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def macro_cluster_bootstrap_draws(
    delta: Sequence[float] | np.ndarray,
    class_ids: Sequence[int] | np.ndarray,
    component_ids: Sequence[str] | np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return bootstrap draws for an equal-class mean of image-level deltas.

    Components are sampled with replacement within each class. The effect for
    each sampled class is sum(sampled image deltas) / number of sampled images,
    followed by an unweighted average over classes.
    """
    d = np.asarray(delta, dtype=np.float64)
    y = np.asarray(class_ids)
    g = np.asarray(component_ids, dtype=str)
    if d.ndim != 1 or y.ndim != 1 or g.ndim != 1 or not (len(d) == len(y) == len(g)):
        raise ValueError("delta, class_ids and component_ids must be aligned 1-D arrays")
    if not len(d) or not np.isfinite(d).all() or replicates < 1:
        raise ValueError("bootstrap requires finite nonempty deltas and replicates >= 1")

    classes = np.unique(y)
    draws = np.zeros(replicates, dtype=np.float64)
    for class_id in classes:
        mask = y == class_id
        class_components = np.unique(g[mask])
        sums = np.empty(len(class_components), dtype=np.float64)
        sizes = np.empty(len(class_components), dtype=np.int64)
        for j, component_id in enumerate(class_components):
            values = d[mask & (g == component_id)]
            sums[j] = values.sum(dtype=np.float64)
            sizes[j] = values.size
        chosen = rng.integers(0, len(class_components), size=(replicates, len(class_components)))
        sampled_sum = sums[chosen].sum(axis=1)
        sampled_size = sizes[chosen].sum(axis=1)
        draws += (sampled_sum / sampled_size) / len(classes)
    return draws


def macro_cluster_bootstrap_ci(
    delta: Sequence[float] | np.ndarray,
    class_ids: Sequence[int] | np.ndarray,
    component_ids: Sequence[str] | np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    draws = macro_cluster_bootstrap_draws(delta, class_ids, component_ids, replicates, rng)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)
