"""
Avalanche effect tests — flipping a single bit in the seed must change
approximately 50% of the output bits. This is a core property of any
good hash-based PRNG and directly tests the diffusion quality of the
underlying algorithm.

These tests will be most meaningful once the real Tyche algorithm
replaces the Threefry stub.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tyche import impl

pytestmark = pytest.mark.statistical

N_TRIALS = 100   # number of bit-flip trials
N_SAMPLES = 32   # uint32 words generated per trial (= 1024 output bits)


def hamming_distance_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of bits that differ between two uint32 arrays."""
    xor = np.bitwise_xor(a, b)
    total_bits = xor.size * 32
    differing = sum(bin(int(x)).count('1') for x in xor.flatten())
    return differing / total_bits


def test_seed_avalanche():
    """Flipping one bit in the seed must change ~50% of the output bits."""
    fractions = []
    base_seed = 42

    for bit in range(32):  # flip each of the first 32 seed bits
        flipped_seed = base_seed ^ (1 << bit)

        key_a = jax.random.key(base_seed, impl=impl)
        key_b = jax.random.key(flipped_seed, impl=impl)

        bits_a = np.array(jax.random.bits(key_a, shape=(N_SAMPLES,), dtype=jnp.uint32))
        bits_b = np.array(jax.random.bits(key_b, shape=(N_SAMPLES,), dtype=jnp.uint32))

        frac = hamming_distance_fraction(bits_a, bits_b)
        fractions.append(frac)

    mean_frac = np.mean(fractions)
    # A good avalanche effect means ~50% bits change, allow 10% tolerance
    assert abs(mean_frac - 0.5) < 0.10, (
        f"Poor avalanche effect: mean bit change fraction = {mean_frac:.3f} (expected ~0.5)"
    )


def test_fold_in_avalanche():
    """Flipping one bit in fold_in data must change ~50% of the output bits."""
    fractions = []
    key = jax.random.key(42, impl=impl)

    for bit in range(16):
        data_a = np.uint32(100)
        data_b = np.uint32(100 ^ (1 << bit))

        key_a = jax.random.fold_in(key, int(data_a))
        key_b = jax.random.fold_in(key, int(data_b))

        bits_a = np.array(jax.random.bits(key_a, shape=(N_SAMPLES,), dtype=jnp.uint32))
        bits_b = np.array(jax.random.bits(key_b, shape=(N_SAMPLES,), dtype=jnp.uint32))

        frac = hamming_distance_fraction(bits_a, bits_b)
        fractions.append(frac)

    mean_frac = np.mean(fractions)
    assert abs(mean_frac - 0.5) < 0.10, (
        f"Poor fold_in avalanche: mean bit change fraction = {mean_frac:.3f} (expected ~0.5)"
    )


def test_split_avalanche():
    """Child keys from different parents must differ in ~50% of output bits."""
    fractions = []

    for i in range(N_TRIALS):
        key_a = jax.random.key(i, impl=impl)
        key_b = jax.random.key(i ^ 1, impl=impl)  # flip one bit

        child_a = jax.random.split(key_a, num=1)[0]
        child_b = jax.random.split(key_b, num=1)[0]

        bits_a = np.array(jax.random.bits(child_a, shape=(N_SAMPLES,), dtype=jnp.uint32))
        bits_b = np.array(jax.random.bits(child_b, shape=(N_SAMPLES,), dtype=jnp.uint32))

        frac = hamming_distance_fraction(bits_a, bits_b)
        fractions.append(frac)

    mean_frac = np.mean(fractions)
    assert abs(mean_frac - 0.5) < 0.10, (
        f"Poor split avalanche: mean bit change fraction = {mean_frac:.3f} (expected ~0.5)"
    )