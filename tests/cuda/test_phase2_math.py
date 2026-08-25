import jax
import jax.numpy as jnp
import pytest
from tyche.config import TycheConfig
import numpy as np

def test_math_oracle_v1():
    # Setup baseline and CUDA configurations
    cfg_jax = TycheConfig(tile_size=16, num_rounds=4, backend="jax", embedding="hash")
    cfg_cuda = TycheConfig(tile_size=16, num_rounds=4, backend="cuda", embedding="hash")
    
    impl_jax = cfg_jax.build()
    impl_cuda = cfg_cuda.build()
    
    # Generate seed
    seed = [42, 1337]
    
    # Generate random bits using JAX (Python implementation oracle)
    key_jax = impl_jax.seed(seed)
    # Fold in some data
    key_jax = impl_jax.fold_in(key_jax, jnp.array(99))
    # Generate 1000 uint32s
    bits_jax = impl_jax.random_bits(key_jax, 32, (1000,))
    
    # Generate random bits using CUDA
    key_cuda = impl_cuda.seed(seed)
    key_cuda = impl_cuda.fold_in(key_cuda, jnp.array(99))
    bits_cuda = impl_cuda.random_bits(key_cuda, 32, (1000,))
    
    # Mathematical Equivalence check
    np.testing.assert_array_equal(np.array(bits_jax), np.array(bits_cuda))
    
    # Split test
    key_jax_split = impl_jax.split(key_jax, (2,))
    key_cuda_split = impl_cuda.split(key_cuda, (2,))
    np.testing.assert_array_equal(np.array(key_jax_split), np.array(key_cuda_split))
    
    # Sub-key bits test
    bits_jax_child = impl_jax.random_bits(key_jax_split[0], 64, (500,))
    bits_cuda_child = impl_cuda.random_bits(key_cuda_split[0], 64, (500,))
    np.testing.assert_array_equal(np.array(bits_jax_child), np.array(bits_cuda_child))

def test_frequencies_v1():
    cfg = TycheConfig(tile_size=16, num_rounds=4, backend="cuda", embedding="hash")
    impl = cfg.build()
    
    key = impl.seed(12345)
    # 1 million uint32s
    bits = impl.random_bits(key, 32, (1_000_000,))
    
    # Simple frequency test - bit population should be ~50%
    popcount = 0
    # Numpy count_nonzero is faster on CPU but let's do a basic bit count
    # Since we can't easily popcount natively in python without loops, we'll just check mean
    # of the bits casted to float.
    mean_val = np.mean(np.array(bits, dtype=np.float64))
    expected = (2**32 - 1) / 2
    # Should be within 0.1% of expected
    assert np.abs(mean_val - expected) / expected < 0.001
