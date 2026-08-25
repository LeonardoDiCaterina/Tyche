import jax.numpy as jnp
from dataclasses import dataclass
from enum import Enum

class EmbeddingType(Enum):
    HASH = 0
    PRNG = 1

@dataclass
class TycheV5bConfig:
    blocks: int
    warps_per_block: int
    T: int
    R: int
    embedding_type: EmbeddingType
    weight_matrices: jnp.ndarray
    key_mix: jnp.ndarray
    
    @classmethod
    def create(cls, 
               keys: jnp.ndarray,
               blocks: int = 108,
               warps_per_block: int = 8,
               T: int = 16,
               R: int = 1,
               embedding_type: EmbeddingType = EmbeddingType.HASH) -> 'TycheV5bConfig':
        assert keys.shape == (8,)
        assert keys.dtype == jnp.uint32
        
        # Tyche V5b only uses 1 weight matrix and 1 key mix
        # W_0
        w_0_scalar = keys[2] ^ keys[3] ^ 0x9E3779B9
        weight_matrices = jnp.full((1, 16), w_0_scalar, dtype=jnp.uint32)
        
        key_mix = jnp.array([keys[0] ^ keys[1]], dtype=jnp.uint32)
        
        return cls(
            blocks=blocks,
            warps_per_block=warps_per_block,
            T=T,
            R=R,
            embedding_type=embedding_type,
            weight_matrices=weight_matrices,
            key_mix=key_mix
        )
