"""
Tests that Tyche operations survive jit compilation.
jit is the most common JAX transformation — if anything breaks here,
it breaks for virtually every real-world user.
"""

import jax
import jax.numpy as jnp
import pytest
from tyche import impl


def make_key(seed=42):
    return jax.random.key(seed, impl=impl)


def test_jit_key_creation():
    """PRNGKey creation must work inside a jitted function."""
    @jax.jit
    def create_key(seed):
        return jax.random.key(seed, impl=impl)

    key = create_key(42)
    assert key.shape == ()


def test_jit_split():
    """split must work inside a jitted function."""
    @jax.jit
    def do_split(key):
        return jax.random.split(key, num=4)

    key = make_key()
    children = do_split(key)
    assert children.shape == (4,)


def test_jit_fold_in():
    """fold_in must work inside a jitted function."""
    @jax.jit
    def do_fold_in(key, data):
        return jax.random.fold_in(key, data)

    key = make_key()
    result = do_fold_in(key, jnp.uint32(7))
    assert result.shape == ()


def test_jit_random_bits():
    """random_bits must work inside a jitted function."""
    @jax.jit
    def do_bits(key):
        return jax.random.bits(key, shape=(64,), dtype=jnp.uint32)

    key = make_key()
    bits = do_bits(key)
    assert bits.shape == (64,)
    assert bits.dtype == jnp.uint32


def test_jit_uniform():
    """jax.random.uniform must work inside a jitted function."""
    @jax.jit
    def do_uniform(key):
        return jax.random.uniform(key, shape=(100,))

    key = make_key()
    samples = do_uniform(key)
    assert samples.shape == (100,)
    assert jnp.all((samples >= 0.0) & (samples < 1.0))


def test_jit_normal():
    """jax.random.normal must work inside a jitted function."""
    @jax.jit
    def do_normal(key):
        return jax.random.normal(key, shape=(100,))

    key = make_key()
    samples = do_normal(key)
    assert samples.shape == (100,)
    assert jnp.all(jnp.isfinite(samples))


def test_jit_determinism():
    """jitted functions must produce the same result across calls."""
    @jax.jit
    def do_uniform(key):
        return jax.random.uniform(key, shape=(50,))

    key = make_key()
    result1 = do_uniform(key)
    result2 = do_uniform(key)
    assert jnp.array_equal(result1, result2)


def test_jit_recompilation_with_different_key():
    """jitted function must give different results for different keys."""
    @jax.jit
    def do_uniform(key):
        return jax.random.uniform(key, shape=(50,))

    key1 = make_key(seed=1)
    key2 = make_key(seed=2)
    assert not jnp.array_equal(do_uniform(key1), do_uniform(key2))