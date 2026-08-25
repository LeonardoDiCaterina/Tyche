"""
TycheConfig — configurable factory for the Tyche PRNGImpl.

Usage:
    from tyche.config import TycheConfig

    cfg = TycheConfig(block_size=4, num_rounds=16)
    impl = cfg.build()
    key = jax.random.PRNGKey(42, impl=impl)"""

import math
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  
from jax._src.random.prng import PRNGImpl

from tyche.algorithm import (
    key_size_uint32,
    expand_seed_to_key,
    make_hash_parallel,
    make_tile,
    _key_to_matrices,
    derive_child_key,
)

class TycheConfig:
    """
    Configuration object for the Tyche PRNG.
    """
    def __init__(self, tile_size: int = 16, num_rounds: int = 4, backend:str = "jax", embedding:str = "hash"):
        
        if backend == "pallas":
            from tyche.backend_pallas import PallasBackend
            self._backend = PallasBackend(num_rounds, tile_size)
        else:
            from tyche.backend_jax import JaxBackend
            self._backend = JaxBackend(num_rounds, tile_size)
        
        if tile_size not in (16, 32, 64):
            raise ValueError(f"tile_size must be one of 16, 32, 64 — got {tile_size}")
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1 — got {num_rounds}")

        self.tile_size = tile_size
        self.num_rounds = num_rounds
        self.embedding = embedding
        self._hash_parallel = make_hash_parallel(num_rounds)

    @property
    def key_shape(self):
        return (key_size_uint32(self.num_rounds, self.tile_size),)

    @property
    def name(self):
        return f"tyche_t{self.tile_size}_r{self.num_rounds}_{self.embedding}"

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
        
        # 1 int8 per byte. 32-bit = 4 bytes. 64-bit = 8 bytes.
        bytes_per_elem = bit_width // 8
        total_bytes = total_out_elems * bytes_per_elem
        num_tiles = math.ceil(total_bytes / (T * T))

        weight_matrices = _key_to_matrices(key, R, T)
        tiles = make_tile(key, 0, num_tiles, T, self.embedding)
        hashed = self._hash_parallel(tiles, weight_matrices)

        flat_i8 = hashed.reshape(-1)[:total_bytes]
        out_dtype = jnp.uint32 if bit_width == 32 else jnp.uint64
        packed = flat_i8.view(out_dtype)

        return packed.reshape(shape)

    def build(self) -> PRNGImpl:
        return PRNGImpl(
            key_shape=self.key_shape,
            seed=self._seed,
            split=self._split,
            fold_in=self._fold_in,
            random_bits=self._random_bits,
            name=self.name,
            tag="tyc",
        )