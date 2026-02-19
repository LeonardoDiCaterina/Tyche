"""
Tests that Tyche produces uniformly distributed outputs.
Uses the Kolmogorov-Smirnov test and chi-square test.

These tests are probabilistic — they can fail by chance with low probability.
Significance level is set conservatively (alpha=0.001) to minimize false failures.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from tyche import impl

pytestmark = pytest.mark.statistical

ALPHA = 0.001  # very conservative — we want near-zero false failure rate


def make_key(seed=42):
    return jax.random.key(seed, impl=impl)


def test_uniform_ks_test(sample_size):
    """KS test: uniform samples must not be distinguishable from U(0,1)."""
    key = make_key()
    samples = np.array(jax.random.uniform(key, shape=(sample_size,)))
    stat, p_value = stats.kstest(samples, "uniform")
    assert p_value > ALPHA, (
        f"KS test failed: p={p_value:.4f} < {ALPHA} (n={sample_size})"
    )


def test_uniform_chi_square(sample_size):
    """Chi-square test: uniform samples must be evenly distributed across bins."""
    key = make_key()
    samples = np.array(jax.random.uniform(key, shape=(sample_size,)))
    n_bins = 20
    observed, _ = np.histogram(samples, bins=n_bins, range=(0, 1))
    expected = np.full(n_bins, sample_size / n_bins)
    stat, p_value = stats.chisquare(observed, expected)
    assert p_value > ALPHA, (
        f"Chi-square test failed: p={p_value:.4f} < {ALPHA} (n={sample_size}, bins={n_bins})"
    )


def test_uniform_mean(sample_size):
    """Mean of uniform samples must be close to 0.5."""
    key = make_key()
    samples = jax.random.uniform(key, shape=(sample_size,))
    mean = float(jnp.mean(samples))
    # 6 sigma tolerance
    tolerance = 6 * (1 / (12 * sample_size)) ** 0.5
    assert abs(mean - 0.5) < tolerance, (
        f"Mean {mean:.4f} too far from 0.5 (tolerance={tolerance:.4f})"
    )


def test_uniform_variance(sample_size):
    """Variance of uniform samples must be close to 1/12."""
    key = make_key()
    samples = jax.random.uniform(key, shape=(sample_size,))
    var = float(jnp.var(samples))
    expected_var = 1 / 12
    tolerance = 0.05  # 5% relative tolerance
    assert abs(var - expected_var) < tolerance * expected_var, (
        f"Variance {var:.4f} too far from {expected_var:.4f}"
    )


def test_uniform_across_independent_keys():
    """Pooled samples from independent keys must also be uniform."""
    keys = jax.random.split(make_key(), num=10)
    all_samples = np.concatenate([
        np.array(jax.random.uniform(k, shape=(1000,))) for k in keys
    ])
    stat, p_value = stats.kstest(all_samples, "uniform")
    assert p_value > ALPHA, (
        f"Pooled KS test failed: p={p_value:.4f} < {ALPHA}"
    )


def test_normal_distribution(sample_size):
    """Normal samples must pass a normality test."""
    key = make_key()
    samples = np.array(jax.random.normal(key, shape=(sample_size,)))
    stat, p_value = stats.normaltest(samples)
    assert p_value > ALPHA, (
        f"Normality test failed: p={p_value:.4f} < {ALPHA}"
    )