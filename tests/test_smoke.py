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


def test_v2_smoke():
    from tyche.v2.config import TycheV2Config
    cfg = TycheV2Config(tile_size=16, num_rounds=2)
    impl_v2 = cfg.build()
    key = jax.random.key(42, impl=impl_v2)
    samples = jax.random.uniform(key, shape=(100,))
    assert samples.shape == (100,)


def test_v2_1_smoke():
    from tyche.v2_1.config import TycheV2_1Config
    cfg = TycheV2_1Config(tile_size=16, num_rounds=2)
    impl_v2_1 = cfg.build()
    key = jax.random.key(42, impl=impl_v2_1)
    samples = jax.random.uniform(key, shape=(100,))
    assert samples.shape == (100,)


def test_v3_philox_smoke():
    from tyche.v3_philox.config import TycheV3_PhiloxConfig
    # Test 32-bit word Philox
    cfg32 = TycheV3_PhiloxConfig(tile_size=4, num_rounds=2, word_size=32, backend="jax")
    impl32 = cfg32.build()
    key32 = jax.random.key(42, impl=impl32)
    samples32 = jax.random.uniform(key32, shape=(100,))
    assert samples32.shape == (100,)
    
    # Test 64-bit word Philox
    cfg64 = TycheV3_PhiloxConfig(tile_size=4, num_rounds=2, word_size=64, backend="jax")
    impl64 = cfg64.build()
    key64 = jax.random.key(42, impl=impl64)
    samples64 = jax.random.uniform(key64, shape=(100,))
    assert samples64.shape == (100,)


def test_v4_threefry_smoke():
    from tyche.v4_threefry.config import TycheV4_ThreefryConfig
    # Test 32-bit word Threefry
    cfg32 = TycheV4_ThreefryConfig(tile_size=4, num_rounds=2, word_size=32, backend="jax")
    impl32 = cfg32.build()
    key32 = jax.random.key(42, impl=impl32)
    samples32 = jax.random.uniform(key32, shape=(100,))
    assert samples32.shape == (100,)

    # Test 64-bit word Threefry
    cfg64 = TycheV4_ThreefryConfig(tile_size=4, num_rounds=2, word_size=64, backend="jax")
    impl64 = cfg64.build()
    key64 = jax.random.key(42, impl=impl64)
    samples64 = jax.random.uniform(key64, shape=(100,))
    assert samples64.shape == (100,)