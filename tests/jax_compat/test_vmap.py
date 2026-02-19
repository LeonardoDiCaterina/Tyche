"""
tests/jax_compat/test_vmap.py

Tests that Tyche operations survive vmap.
vmap is how JAX users vectorize over batches of keys — extremely common
in ML training loops (per-sample randomness, ensemble models, etc.)
"""

import jax
import jax.numpy as jnp
import pytest
from tyche import impl


def make_key(seed=42):
    return jax.random.key(seed, impl=impl)


def test_vmap_over_split_keys():
    """Applying a random function over a batch of split keys must work."""
    key = make_key()
    keys = jax.random.split(key, num=8)

    def sample(k):
        return jax.random.uniform(k, shape=(10,))

    batched_sample = jax.vmap(sample)
    result = batched_sample(keys)
    assert result.shape == (8, 10)


def test_vmap_produces_different_samples_per_key():
    """Each key in the batch must produce different samples."""
    key = make_key()
    keys = jax.random.split(key, num=4)

    batched_sample = jax.vmap(lambda k: jax.random.uniform(k, shape=(20,)))
    result = batched_sample(keys)

    for i in range(result.shape[0]):
        for j in range(i + 1, result.shape[0]):
            assert not jnp.array_equal(result[i], result[j]), (
                f"vmap keys {i} and {j} produced identical samples"
            )


def test_vmap_over_fold_in():
    """fold_in vectorized over a range of data values must work."""
    key = make_key()
    data = jnp.arange(8, dtype=jnp.uint32)

    folded = jax.vmap(lambda d: jax.random.fold_in(key, d))(data)
    assert folded.shape == (8,)


def test_vmap_fold_in_produces_distinct_keys():
    """Each fold_in result in the vmap batch must be distinct."""
    key = make_key()
    data = jnp.arange(8, dtype=jnp.uint32)

    folded = jax.vmap(lambda d: jax.random.fold_in(key, d))(data)
    for i in range(len(folded)):
        for j in range(i + 1, len(folded)):
            assert not jnp.array_equal(folded[i], folded[j])


def test_vmap_normal():
    """jax.random.normal must work inside vmap."""
    key = make_key()
    keys = jax.random.split(key, num=16)

    batched_normal = jax.vmap(lambda k: jax.random.normal(k, shape=(50,)))
    result = batched_normal(keys)

    assert result.shape == (16, 50)
    assert jnp.all(jnp.isfinite(result))


def test_vmap_and_jit_compose():
    """vmap and jit must compose correctly together."""
    key = make_key()
    keys = jax.random.split(key, num=8)

    @jax.jit
    @jax.vmap
    def batched_uniform(k):
        return jax.random.uniform(k, shape=(32,))

    result = batched_uniform(keys)
    assert result.shape == (8, 32)
    assert jnp.all((result >= 0.0) & (result < 1.0))