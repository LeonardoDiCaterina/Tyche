"""
Adversarial chosen-input statistical tests.

These exercises target edge-case key/seed patterns that could reveal
weaknesses in the key schedule or seed expansion.  They are intentionally
structured rather than purely random.

1. Related-key XOR test: create two streams from keys that differ in exactly
   one bit and verify that their XOR behaves like a random stream.
2. Zero-/one-/alternating-heavy keys: use degenerate seeds such as all-zeros,
   all-ones, alternating bits, etc., and check that output still meets simple
   bit-balance expectations.
3. Sequential seeds interleaved: generate streams from a sequence of nearby
   seeds, interleave their outputs, and check for cross-seed correlations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from tyche import impl

pytestmark = pytest.mark.statistical

# reuse size constant from other tests; keep moderate to avoid long runtimes
N = 10_000


def bit_balance_fraction(stream: np.ndarray, bit_pos: int) -> float:
    ones = int(np.sum((stream >> bit_pos) & 1))
    return ones / stream.size


def check_bit_balance(stream: np.ndarray, tol: float = 0.03):
    for bit in range(32):
        frac = bit_balance_fraction(stream, bit)
        assert abs(frac - 0.5) < tol, f"Bit {bit} has imbalance {frac:.3f}"


def test_related_key_xor():
    """XOR of streams from keys differing by one bit should be uniformly
    balanced across all bit positions.
    """
    base_seed = 0x12345678
    base_key = jax.random.key(base_seed, impl=impl)
    base_stream = np.array(
        jax.random.bits(base_key, shape=(N,), dtype=jnp.uint32)
    )

    for bit in range(32):
        flipped_seed = base_seed ^ (1 << bit)
        other_key = jax.random.key(flipped_seed, impl=impl)
        other_stream = np.array(
            jax.random.bits(other_key, shape=(N,), dtype=jnp.uint32)
        )
        xor_stream = base_stream ^ other_stream
        # each bit in the XOR should be ~50% ones
        check_bit_balance(xor_stream, tol=0.03)


def test_zero_heavy_keys():
    """Keys generated from degenerately-structured seeds must still produce
    balanced output.
    """
    seeds = [0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555]
    for seed in seeds:
        stream = np.array(
            jax.random.bits(jax.random.key(seed, impl=impl),
                             shape=(N,), dtype=jnp.uint32)
        )
        check_bit_balance(stream)


def test_sequential_seeds_interleaved():
    """Interleave outputs from seeds 0,1,2,... and check for balance and low
    correlation between adjacent-seed streams.
    """
    num_seeds = 8
    seeds = list(range(num_seeds))
    streams = []
    for s in seeds:
        streams.append(
            np.array(
                jax.random.bits(jax.random.key(s, impl=impl),
                                 shape=(N,), dtype=jnp.uint32)
            )
        )
    # pairwise correlation of neighbouring streams
    for i in range(num_seeds - 1):
        corr = np.corrcoef(streams[i].astype(np.float64),
                           streams[i + 1].astype(np.float64))[0, 1]
        assert abs(corr) < 0.05, (
            f"Correlation between seed {i} and {i+1} too high: r={corr:.4f}"
        )

    # form interleaved sequence and check bit balance across the whole block
    interleaved = np.empty(N * num_seeds, dtype=np.uint32)
    for idx, st in enumerate(streams):
        interleaved[idx::num_seeds] = st
    check_bit_balance(interleaved, tol=0.02)


def test_sequential_seeds_uniformity():
    """Chi-square test on interleaved streams to look for non-uniformity."""
    num_seeds = 4
    Nsmall = 2000
    seeds = list(range(num_seeds))
    inter = []
    for s in seeds:
        inter.append(
            np.array(
                jax.random.bits(jax.random.key(s, impl=impl),
                                 shape=(Nsmall,), dtype=jnp.uint32)
            )
        )
    interleaved = np.empty(Nsmall * num_seeds, dtype=np.uint32)
    for idx, st in enumerate(inter):
        interleaved[idx::num_seeds] = st
    # test low 8 bits distribution
    vals = interleaved & 0xFF
    observed, _ = np.histogram(vals, bins=256, range=(0, 256))
    expected = np.full(256, len(vals) / 256)
    stat, p = stats.chisquare(observed, expected)
    assert p > 0.001, f"Sequential-seed interleaved uniformity failed: p={p:.4f}"
