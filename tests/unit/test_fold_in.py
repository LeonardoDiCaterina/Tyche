"""
Contract tests for the fold_in function.
A valid fold_in implementation must:
- Return a key of the same shape as the input
- Be deterministic (same key + same data → same result)
- Be sensitive to data (different data → different output keys)
- Be sensitive to the key (different keys + same data → different outputs)
- Not return the original key unchanged
"""

import jax
import jax.numpy as jnp
import pytest
from tyche import impl


def make_key(seed=42):
    return jax.random.key(seed, impl=impl)


def test_fold_in_output_shape():
    """fold_in must return a key with the same shape as the input."""
    key = make_key()
    result = jax.random.fold_in(key, 0)
    assert result.shape == ()


def test_fold_in_is_deterministic():
    """Same key and data must always produce the same output."""
    key = make_key()
    result1 = jax.random.fold_in(key, 7)
    result2 = jax.random.fold_in(key, 7)
    assert jnp.array_equal(result1, result2)


def test_fold_in_sensitive_to_data():
    """Different data values must produce different output keys."""
    key = make_key()
    results = [jax.random.fold_in(key, d) for d in range(16)]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not jnp.array_equal(results[i], results[j]), (
                f"fold_in with data={i} and data={j} produced the same key"
            )


def test_fold_in_sensitive_to_key():
    """Same data folded into different keys must produce different outputs."""
    keys = [jax.random.key(s, impl=impl) for s in [0, 1, 42, 99]]
    results = [jax.random.fold_in(k, 7) for k in keys]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not jnp.array_equal(results[i], results[j]), (
                f"Different parent keys produced the same fold_in result"
            )


def test_fold_in_changes_key():
    """fold_in must not return the original key unchanged."""
    key = make_key()
    result = jax.random.fold_in(key, 1)
    assert not jnp.array_equal(key, result)


@pytest.mark.parametrize("data", [0, 1, 255, 2**16 - 1, 2**32 - 1])
def test_fold_in_edge_case_data(data):
    """Edge case data values must not crash and return valid keys."""
    key = make_key()
    result = jax.random.fold_in(key, data)
    assert result.shape == ()