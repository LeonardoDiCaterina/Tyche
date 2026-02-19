"""
tests/test_smoke.py

Minimal smoke tests — confirms the infrastructure is wired up correctly.
If these pass, Phase 1 is done.
"""

import jax
import jax.numpy as jnp
from tyche import impl


def test_impl_is_registered():
    """Tyche impl object exists and has the expected name."""
    assert impl is not None
    assert impl.name.startswith("tyche")


def test_prng_key_creation():
    """Can create a key using the Tyche impl without errors."""
    key = jax.random.key(42, impl=impl)
    assert key is not None


def test_key_shape():
    """Key has the shape declared in the PRNGImpl."""
    key = jax.random.key(42, impl=impl)
    # New-style typed keys have scalar shape; internal data matches key_shape
    assert key.shape == ()


def test_split_produces_keys():
    """split() returns the requested number of keys."""
    key = jax.random.key(42, impl=impl)
    keys = jax.random.split(key, num=4)
    assert keys.shape[0] == 4


def test_random_uniform_runs():
    """jax.random.uniform works end-to-end with the Tyche impl."""
    key = jax.random.key(42, impl=impl)
    samples = jax.random.uniform(key, shape=(100,))
    assert samples.shape == (100,)
    assert jnp.all((samples >= 0.0) & (samples < 1.0))