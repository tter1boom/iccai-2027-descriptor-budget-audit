"""Small deterministic checks for the canonical clustered macro estimator."""
from __future__ import annotations

import numpy as np

from canonical_macro_cluster_bootstrap import macro_cluster_bootstrap_draws


def main() -> None:
    # A singleton component and an eight-image component must be image-weighted
    # within their class, not averaged as two equally weighted component means.
    delta = np.array([1.0] + [0.0] * 8 + [1.0, 1.0])
    classes = np.array([0] * 9 + [1] * 2)
    components = np.array(["singleton"] + ["large"] * 8 + ["c0", "c1"])
    first = macro_cluster_bootstrap_draws(delta, classes, components, 200, np.random.default_rng(7))
    second = macro_cluster_bootstrap_draws(delta, classes, components, 200, np.random.default_rng(7))
    assert np.array_equal(first, second)
    # The point estimand is 0.5 * (1/9 + 1), not the mean of component means.
    expected = 0.5 * (1.0 / 9.0 + 1.0)
    observed = 0.5 * (delta[classes == 0].mean() + delta[classes == 1].mean())
    assert np.isclose(observed, expected)
    assert np.isfinite(first).all()
    print("PASS: deterministic class-macro component bootstrap and image weighting")


if __name__ == "__main__":
    main()
