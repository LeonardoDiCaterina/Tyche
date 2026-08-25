import math
import jax
import jax.numpy as jnp
try:
    from jax._src.random.prng import PRNGImpl
except ImportError:
    from jax._src.prng import PRNGImpl

from tyche.v2.algorithm import (
    key_size_uint32,
    expand_seed_to_key,
    _key_to_matrices,
    derive_child_key,
)

class TycheV5bConfig:
    def __init__(self, tile_size: int = 16, num_rounds: int = 1, embedding: str = "hash", backend: str = "cuda"):
        if backend != "cuda":
            raise ValueError("Tyche V5b Hybrid only supports backend='cuda'!")
        if tile_size != 16:
            raise ValueError("Tyche V5b Hybrid MVP only supports tile_size=16")
        if num_rounds != 1:
            raise ValueError("Tyche V5b Hybrid MVP only supports num_rounds=1")

        from tyche.v5b_hybrid.backend_cuda import CudaBackendV5b
        self._backend = CudaBackendV5b(num_rounds, tile_size)
        
        self.tile_size = tile_size
        self.num_rounds = num_rounds
        self.embedding = embedding

    @property
    def key_shape(self):
        return (key_size_uint32(self.num_rounds, self.tile_size),)

    @property
    def name(self):
        return f"tyc5_hybrid_t{self.tile_size}_{self.embedding}"

    def _seed(self, seed) -> jnp.ndarray:
        return expand_seed_to_key(seed, self.num_rounds, self.tile_size)

    def _split(self, key: jnp.ndarray, shape: tuple) -> jnp.ndarray:
        num = math.prod(shape)
        child_indices = jnp.arange(num, dtype=jnp.uint64)
        children = jax.vmap(
            lambda i: derive_child_key(key, i, self.num_rounds, self.tile_size)
        )(child_indices)
        return children.reshape(*shape, *self.key_shape)

    def _fold_in(self, key: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        return derive_child_key(
            key, data.astype(jnp.uint64), self.num_rounds, self.tile_size
        )

    def _random_bits(self, key: jnp.ndarray, bit_width: int, shape: tuple) -> jnp.ndarray:
        T, R = self.tile_size, self.num_rounds

        total_out_elems = math.prod(shape)
        # We need this many uint32 outputs (if bit_width=32)
        u32_per_elem = 1 if bit_width == 32 else 2
        total_u32_needed = total_out_elems * u32_per_elem
        
        # V5b outputs uint32 directly! 
        u32_per_tile = T * T
        num_tiles = math.ceil(total_u32_needed / u32_per_tile)

        weight_matrices = _key_to_matrices(key, R, T)
        
        # Returns (num_tiles, T, T) of uint32
        hashed_tiles = self._backend.hash_parallel(key, 0, num_tiles, weight_matrices, self.embedding)

        flat_u32 = hashed_tiles.reshape(-1)[:total_u32_needed]
        
        if bit_width == 64:
            return flat_u32.view(jnp.uint64).reshape(shape)
        return flat_u32.reshape(shape)

    def build(self) -> PRNGImpl:
        return PRNGImpl(
            key_shape=self.key_shape,
            seed=self._seed,
            split=self._split,
            fold_in=self._fold_in,
            random_bits=self._random_bits,
            name=self.name,
            tag="tyc5",
        )
