"""
Bit-level statistical tests on raw output from random_bits.
A good PRNG must have:
- ~50% ones in every bit position
- Balanced byte distribution
- No suspicious patterns in runs of identical bits
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from tyche import impl

pytestmark = pytest.mark.statistical

N = 10_000  # number of uint32 samples


def get_bits(seed=42):
    key = jax.random.key(seed, impl=impl)
    return np.array(jax.random.bits(key, shape=(N,), dtype=jnp.uint32))


def test_bit_balance():
    """Each of the 32 bit positions must have ~50% ones."""
    samples = get_bits()
    for bit_pos in range(32):
        ones = int(np.sum((samples >> bit_pos) & 1))
        fraction = ones / N
        # Allow 3% deviation from 50%
        assert abs(fraction - 0.5) < 0.03, (
            f"Bit {bit_pos}: {fraction:.3f} ones (expected ~0.5)"
        )


def test_byte_distribution():
    """Each byte of the output must be roughly uniformly distributed."""
    samples = get_bits()
    for byte_pos in range(4):
        byte_vals = (samples >> (byte_pos * 8)) & 0xFF
        observed, _ = np.histogram(byte_vals, bins=256, range=(0, 256))
        expected = np.full(256, N / 256)
        stat, p_value = stats.chisquare(observed, expected)
        assert p_value > 0.001, (
            f"Byte {byte_pos} chi-square failed: p={p_value:.4f}"
        )


def test_no_runs_of_identical_values():
    """Must not have suspiciously long runs of identical uint32 values."""
    samples = get_bits()
    max_run = 1
    current_run = 1
    for i in range(1, len(samples)):
        if samples[i] == samples[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    # A run of 5+ identical uint32s in 10k samples is astronomically unlikely
    assert max_run < 5, f"Suspicious run of {max_run} identical uint32 values"


def test_msb_balance():
    """Most significant bit must be ~50% ones."""
    samples = get_bits()
    msb_ones = int(np.sum(samples >> 31))
    fraction = msb_ones / N
    assert abs(fraction - 0.5) < 0.03, (
        f"MSB fraction: {fraction:.3f} (expected ~0.5)"
    )


def test_lsb_balance():
    """Least significant bit must be ~50% ones."""
    samples = get_bits()
    lsb_ones = int(np.sum(samples & 1))
    fraction = lsb_ones / N
    assert abs(fraction - 0.5) < 0.03, (
        f"LSB fraction: {fraction:.3f} (expected ~0.5)"
    )


def test_bits_from_independent_seeds_differ():
    """Bit streams from different seeds must not be correlated."""
    bits_a = get_bits(seed=1).astype(np.float64)
    bits_b = get_bits(seed=2).astype(np.float64)
    correlation = np.corrcoef(bits_a, bits_b)[0, 1]
    assert abs(correlation) < 0.05, (
        f"Suspicious correlation between bit streams: r={correlation:.4f}"
    )