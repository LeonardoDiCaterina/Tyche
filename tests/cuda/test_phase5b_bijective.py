import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tyche.v5b_bijective import generate_v5b_cuda, TycheV5bConfig, EmbeddingType

@pytest.fixture
def base_keys():
    return jnp.array([
        0x12345678, 0x9ABCDEF0, 0x11223344, 0x55667788,
        0xAABBCCDD, 0xEEFF0011, 0x22334455, 0x66778899
    ], dtype=jnp.uint32)

def test_frequencies_v5b(base_keys):
    config = TycheV5bConfig.create(
        keys=base_keys,
        blocks=108,
        warps_per_block=8,
        T=16,
        R=1,
        embedding_type=EmbeddingType.HASH
    )
    
    out = generate_v5b_cuda(base_keys, config)
    assert out.dtype == jnp.uint32
    
    # We process 108 * 8 * 2 = 1728 tiles = 442368 elements
    assert out.shape == (442368,)
    
    unique, counts = np.unique(out, return_counts=True)
    # The output space is 2^32, so we expect very few duplicates in 442K samples
    assert len(unique) > 440000

def test_avalanche_v5b(base_keys):
    config = TycheV5bConfig.create(
        keys=base_keys,
        blocks=1,
        warps_per_block=1,
        T=16,
        R=1,
        embedding_type=EmbeddingType.HASH
    )
    
    # Generate baseline
    out1 = generate_v5b_cuda(base_keys, config)
    
    # Flip exactly 1 bit in the key
    flipped_keys = base_keys.at[0].set(base_keys[0] ^ 1)
    
    # Recreate config so key_mix changes
    config_flipped = TycheV5bConfig.create(
        keys=flipped_keys,
        blocks=1,
        warps_per_block=1,
        T=16,
        R=1,
        embedding_type=EmbeddingType.HASH
    )
    
    out2 = generate_v5b_cuda(flipped_keys, config_flipped)
    
    diff = out1 ^ out2
    bits_flipped = np.sum([bin(x).count('1') for x in diff])
    total_bits = out1.size * 32
    
    # We expect roughly 50% of bits to flip
    ratio = bits_flipped / total_bits
    assert 0.49 < ratio < 0.51, f"Avalanche failed: {ratio:.3f}"
