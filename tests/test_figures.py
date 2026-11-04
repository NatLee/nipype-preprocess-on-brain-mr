"""Static figure helpers (the figures themselves are drawn in test_python_nodes.py)."""

import numpy as np

from analysis.figures import histogram_edges


def test_histogram_bins_hold_the_same_number_of_integer_levels():
    integers = np.arange(0, 500, dtype=float)
    edges = histogram_edges(integers, scale=300.0)
    counts, _ = np.histogram(integers / 300.0, bins=edges)
    assert len(set(counts[:-1])) == 1  # every full bin holds the same number of levels
    floats = np.linspace(0, 1, 50) + 0.001
    assert len(histogram_edges(floats, scale=1.0)) == 129
