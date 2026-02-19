"""
Contract tests for random_bits — the core generation function.
A valid random_bits implementation must:
- Return arrays of the correct shape and dtype
- Support both 32-bit and 64-bit output
- Be deterministic (same key → same bits)
- Be sensitive to the key (different keys → different bits)
- Produce no NaN/Inf (it outputs integers, but we check for all-zeros/all-ones)
"""

import jax
import jax.numpy as jnp
import pytest
from tyche import impl


def make_key(seed=42):
    return jax.random.key(seed, impl=impl)


@pytest.mark.parametrize("shape", [(10,), (4, 4), (2, 3, 5)])
def test_random_bits_output_shape(shape):
    """random_bits must return an array of exactly the requested shape."""
    key = make_key()
    samples = jax.random.bits(key, shape=shape)
    assert samples.shape == shape


def test_random_bits_32_dtype():
    """32-bit mode must return uint32."""
    key = make_key()
    samples = jax.random.bits(key, shape=(100,), dtype=jnp.uint32)
    assert samples.dtype == jnp.uint32


def test_random_bits_is_deterministic():
    """Same key must always produce the same bits."""
    key = make_key()
    bits1 = jax.random.bits(key, shape=(100,))
    bits2 = jax.random.bits(key, shape=(100,))
    assert jnp.array_equal(bits1, bits2)


def test_random_bits_sensitive_to_key():
    """Different keys must produce different bits."""
    key1 = make_key(seed=1)
    key2 = make_key(seed=2)
    bits1 = jax.random.bits(key1, shape=(100,))
    bits2 = jax.random.bits(key2, shape=(100,))
    assert not jnp.array_equal(bits1, bits2)


def test_random_bits_not_constant():
    """Output must not be all-zeros or all-ones (degenerate output)."""
    key = make_key()
    bits = jax.random.bits(key, shape=(256,), dtype=jnp.uint32)
    assert not jnp.all(bits == 0), "random_bits returned all zeros"
    assert not jnp.all(bits == jnp.array(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32)), "random_bits returned all ones"


def test_random_bits_child_keys_produce_different_bits():
    """Each child key from split must produce different bits."""
    key = make_key()
    children = jax.random.split(key, num=8)
    outputs = [jax.random.bits(c, shape=(32,)) for c in children]
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not jnp.array_equal(outputs[i], outputs[j]), (
                f"Children {i} and {j} produced identical bits"
            )


@pytest.mark.parametrize("size", [1, 10, 1000])
def test_random_bits_various_sizes(size):
    """random_bits must handle any size without errors."""
    key = make_key()
    bits = jax.random.bits(key, shape=(size,))
    assert bits.shape == (size,)