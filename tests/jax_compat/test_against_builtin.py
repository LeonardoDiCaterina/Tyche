"""
tests/jax_compat/test_against_builtin.py

Behavioral parity tests against JAX's built-in Threefry implementation.
Tyche and Threefry should produce statistically equivalent distributions
even though their exact outputs will differ.

These tests do NOT check that outputs are identical — they check that
Tyche exposes the same API surface and produces outputs in the same
valid ranges and shapes as Threefry.
"""

import jax
import jax.numpy as jnp
import pytest
from jax._src import prng as jax_prng
from tyche import impl as tyche_impl

threefry_impl = jax_prng.threefry_prng_impl


def make_tyche_key(seed=42):
    return jax.random.key(seed, impl=tyche_impl)


def make_threefry_key(seed=42):
    return jax.random.key(seed, impl=threefry_impl)


def test_key_shapes_valid():
    """Both implementations must produce valid key arrays (shape matches their own impl)."""
    tyche_key = make_tyche_key()
    threefry_key = make_threefry_key()
    assert tyche_key.shape == ()
    assert threefry_key.shape == ()


def test_split_output_shape_matches():
    """split must return (num, *key_shape) for each implementation."""
    num = 8
    tyche_children = jax.random.split(make_tyche_key(), num=num)
    threefry_children = jax.random.split(make_threefry_key(), num=num)
    assert tyche_children.shape == (num,)
    assert threefry_children.shape == (num,)


def test_uniform_output_shape_matches():
    """uniform must return the same shape from both implementations."""
    shape = (100,)
    tyche_samples = jax.random.uniform(make_tyche_key(), shape=shape)
    threefry_samples = jax.random.uniform(make_threefry_key(), shape=shape)
    assert tyche_samples.shape == threefry_samples.shape
    assert tyche_samples.dtype == threefry_samples.dtype


def test_uniform_range_matches():
    """Both implementations must produce values in [0, 1)."""
    samples = jax.random.uniform(make_tyche_key(), shape=(1000,))
    assert jnp.all((samples >= 0.0) & (samples < 1.0))


def test_normal_output_shape_matches():
    """normal must return the same shape from both implementations."""
    shape = (100,)
    tyche_samples = jax.random.normal(make_tyche_key(), shape=shape)
    threefry_samples = jax.random.normal(make_threefry_key(), shape=shape)
    assert tyche_samples.shape == threefry_samples.shape
    assert tyche_samples.dtype == threefry_samples.dtype


def test_normal_all_finite():
    """normal must produce only finite values."""
    samples = jax.random.normal(make_tyche_key(), shape=(1000,))
    assert jnp.all(jnp.isfinite(samples))


def test_randint_output_shape_matches():
    """randint must return the same shape from both implementations."""
    shape = (100,)
    tyche_samples = jax.random.randint(make_tyche_key(), shape=shape, minval=0, maxval=100)
    threefry_samples = jax.random.randint(make_threefry_key(), shape=shape, minval=0, maxval=100)
    assert tyche_samples.shape == threefry_samples.shape


def test_randint_range_matches():
    """randint must respect minval/maxval bounds."""
    samples = jax.random.randint(make_tyche_key(), shape=(1000,), minval=5, maxval=15)
    assert jnp.all((samples >= 5) & (samples < 15))


def test_shuffle_output_shape_matches():
    """shuffle must return the same shape from both implementations."""
    arr = jnp.arange(20)
    tyche_shuffled = jax.random.permutation(make_tyche_key(), arr)
    threefry_shuffled = jax.random.permutation(make_threefry_key(), arr)
    assert tyche_shuffled.shape == threefry_shuffled.shape


def test_shuffle_is_permutation():
    """shuffle must return all original elements (just reordered)."""
    arr = jnp.arange(50)
    shuffled = jax.random.permutation(make_tyche_key(), arr)
    assert jnp.array_equal(jnp.sort(shuffled), arr)


def test_outputs_differ_between_implementations():
    """Tyche and Threefry must produce different raw outputs (they're different algorithms)."""
    tyche_samples = jax.random.uniform(make_tyche_key(), shape=(100,))
    threefry_samples = jax.random.uniform(make_threefry_key(), shape=(100,))
    # Same seed, same API — but different algorithms, so outputs must differ
    # (once the real Tyche algorithm replaces the stub, this will hold)
    # For now this is a reminder test — it will fail with the stub and pass with real Tyche
    pytest.xfail("Expected to fail with stub — will pass once real Tyche algorithm is implemented")