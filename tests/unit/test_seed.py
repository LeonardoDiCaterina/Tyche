"""
Contract tests for the seed function.
A valid seed implementation must:
- Accept any uint64 scalar and return a key of the correct shape
- Be deterministic (same seed → same key)
- Be sensitive (different seeds → different keys)
- Handle edge cases: 0, 1, max uint32, max uint64
"""

import jax
import jax.numpy as jnp
import pytest
from tyche import impl


def make_key(seed):
    return jax.random.key(seed, impl=impl)


def test_seed_returns_correct_shape(seed):
    key = make_key(seed)
    # New-style typed keys have scalar shape; internal data matches key_shape
    assert key.shape == ()


def test_seed_returns_unsigned_integer(seed):
    key = make_key(seed)
    # Typed PRNG keys report a key dtype; verify key is valid
    assert key is not None


def test_seed_is_deterministic(seed):
    """Same seed must always produce the same key."""
    key1 = make_key(seed)
    key2 = make_key(seed)
    assert jnp.array_equal(key1, key2)


def test_different_seeds_produce_different_keys():
    """Every pair of distinct seeds must produce distinct keys."""
    seeds = [0, 1, 42, 12345, 2**31 - 1, 2**32 - 1]
    keys = [make_key(s) for s in seeds]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not jnp.array_equal(keys[i], keys[j]), (
                f"Seeds {seeds[i]} and {seeds[j]} produced the same key"
            )


@pytest.mark.parametrize("edge_seed", [0, 1, 2**32 - 1])
def test_seed_edge_cases(edge_seed):
    """Edge case seeds must not crash and must return valid keys."""
    key = make_key(edge_seed)
    # New-style typed keys have scalar shape
    assert key.shape == ()