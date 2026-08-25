import jax
import jax.numpy as jnp
import pytest
from tyche.v5_hybrid.config import TycheV5Config
import numpy as np

def test_frequencies_v5():
    cfg = TycheV5Config(tile_size=16, num_rounds=1, backend="cuda", embedding="hash")
    impl = cfg.build()
    
    key = impl.seed(12345)
    bits = impl.random_bits(key, 32, (1_000_000,))
    
    mean_val = np.mean(np.array(bits, dtype=np.float64))
    expected = (2**32 - 1) / 2
    assert np.abs(mean_val - expected) / expected < 0.01

def test_avalanche_v5():
    cfg = TycheV5Config(tile_size=16, num_rounds=1, backend="cuda", embedding="hash")
    impl = cfg.build()
    
    # Base key
    key1 = impl.seed(12345)
    
    # Key with 1 bit flipped
    # Seed expander uses SplitMix64, so to test kernel avalanche properly, 
    # we should fold_in a 1-bit difference.
    key1 = impl.fold_in(key1, jnp.array(0))
    key2 = impl.fold_in(key1, jnp.array(1)) # 1 bit difference in fold data
    
    bits1 = impl.random_bits(key1, 32, (256,)) # 1 tile
    bits2 = impl.random_bits(key2, 32, (256,)) # 1 tile
    
    # Calculate bit differences
    diff = np.array(bits1) ^ np.array(bits2)
    bit_flips = 0
    for val in diff:
        bit_flips += bin(val).count("1")
        
    total_bits = 256 * 32
    flip_ratio = bit_flips / total_bits
    
    # Strict Avalanche Criterion requires ~50% bit flips
    assert 0.45 < flip_ratio < 0.55
