"""
Tests for independence between samples — a PRNG that passes uniformity
tests can still fail if its outputs are correlated.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from tyche import impl

pytestmark = pytest.mark.statistical

N = 10_000


def make_key(seed=42):
    return jax.random.key(seed, impl=impl)


def test_serial_autocorrelation():
    """Consecutive samples must not be correlated (lag-1 autocorrelation)."""
    key = make_key()
    samples = np.array(jax.random.uniform(key, shape=(N,)))
    correlation = np.corrcoef(samples[:-1], samples[1:])[0, 1]
    assert abs(correlation) < 0.05, (
        f"Lag-1 autocorrelation too high: r={correlation:.4f}"
    )


def test_higher_lag_autocorrelation():
    """Must have low autocorrelation at several lag values."""
    key = make_key()
    samples = np.array(jax.random.uniform(key, shape=(N,)))
    for lag in [2, 4, 8, 16, 32]:
        correlation = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
        assert abs(correlation) < 0.05, (
            f"Lag-{lag} autocorrelation too high: r={correlation:.4f}"
        )


def test_split_child_independence():
    """Samples from sibling keys (same parent split) must be uncorrelated."""
    key = make_key()
    child1, child2 = jax.random.split(key, num=2)
    samples1 = np.array(jax.random.uniform(child1, shape=(N,)))
    samples2 = np.array(jax.random.uniform(child2, shape=(N,)))
    correlation = np.corrcoef(samples1, samples2)[0, 1]
    assert abs(correlation) < 0.05, (
        f"Sibling key correlation too high: r={correlation:.4f}"
    )


def test_fold_in_independence():
    """Samples from fold_in with different data must be uncorrelated."""
    key = make_key()
    key1 = jax.random.fold_in(key, 0)
    key2 = jax.random.fold_in(key, 1)
    samples1 = np.array(jax.random.uniform(key1, shape=(N,)))
    samples2 = np.array(jax.random.uniform(key2, shape=(N,)))
    correlation = np.corrcoef(samples1, samples2)[0, 1]
    assert abs(correlation) < 0.05, (
        f"fold_in key correlation too high: r={correlation:.4f}"
    )


def test_runs_test():
    """Wald-Wolfowitz runs test: samples must not have non-random runs."""
    key = make_key()
    samples = np.array(jax.random.uniform(key, shape=(N,)))
    median = np.median(samples)
    # Convert to binary: above/below median
    binary = (samples > median).astype(int)
    # Count runs
    runs = 1 + np.sum(binary[1:] != binary[:-1])
    n1 = np.sum(binary)
    n2 = N - n1
    # Expected runs and variance under H0
    expected_runs = (2 * n1 * n2) / N + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - N)) / (N ** 2 * (N - 1))
    z = (runs - expected_runs) / np.sqrt(var_runs)
    # Two-tailed z-test at alpha=0.001 → |z| < 3.29
    assert abs(z) < 3.29, (
        f"Runs test failed: z={z:.4f} (expected |z| < 3.29)"
    )


def test_2d_uniformity():
    """Pairs of consecutive samples must be uniform over the unit square."""
    key = make_key()
    samples = np.array(jax.random.uniform(key, shape=(N,)))
    x, y = samples[::2], samples[1::2]
    # Chi-square on a 10x10 grid
    n_bins = 10
    observed, _, _ = np.histogram2d(x, y, bins=n_bins, range=[[0, 1], [0, 1]])
    expected = len(x) / (n_bins ** 2)
    stat, p_value = stats.chisquare(observed.flatten(), np.full(n_bins**2, expected))
    assert p_value > 0.001, (
        f"2D uniformity chi-square failed: p={p_value:.4f}"
    )