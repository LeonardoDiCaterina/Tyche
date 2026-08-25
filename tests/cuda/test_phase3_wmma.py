import jax
import jax.numpy as jnp
import pytest
from tyche.v2.config import TycheV2Config
import numpy as np

def test_wmma_oracle_v2():
    # Setup baseline and CUDA configurations
    cfg_jax = TycheV2Config(tile_size=16, num_rounds=4, backend="jax", embedding="hash")
    cfg_cuda = TycheV2Config(tile_size=16, num_rounds=4, backend="cuda", embedding="hash")
    
    impl_jax = cfg_jax.build()
    impl_cuda = cfg_cuda.build()
    
    # Generate seed
    seed = [42, 1337]
    
    # Generate random bits using JAX (Python implementation oracle)
    key_jax = impl_jax.seed(seed)
    key_jax = impl_jax.fold_in(key_jax, jnp.array(99))
    bits_jax = impl_jax.random_bits(key_jax, 32, (1000,))
    
    # Generate random bits using CUDA (WMMA Tensor Cores)
    key_cuda = impl_cuda.seed(seed)
    key_cuda = impl_cuda.fold_in(key_cuda, jnp.array(99))
    bits_cuda = impl_cuda.random_bits(key_cuda, 32, (1000,))
    
    # Mathematical Equivalence check
    np.testing.assert_array_equal(np.array(bits_jax), np.array(bits_cuda))
    
    # Split test
    key_jax_split = impl_jax.split(key_jax, (2,))
    key_cuda_split = impl_cuda.split(key_cuda, (2,))
    np.testing.assert_array_equal(np.array(key_jax_split), np.array(key_cuda_split))

def test_frequencies_v2():
    cfg = TycheV2Config(tile_size=16, num_rounds=4, backend="cuda", embedding="hash")
    impl = cfg.build()
    
    key = impl.seed(12345)
    bits = impl.random_bits(key, 32, (1_000_000,))
    
    mean_val = np.mean(np.array(bits, dtype=np.float64))
    expected = (2**32 - 1) / 2
    assert np.abs(mean_val - expected) / expected < 0.001
