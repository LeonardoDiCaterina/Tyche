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
from jax._src.prng import PRNGImpl

from tyche.algorithm import (
    key_size_uint32,
    expand_seed_to_key,
    make_hash_parallel,
    make_counter_blocks,
    _key_to_matrices,
    derive_child_key,
)

class TycheConfig:
    """
    Configuration object for the Tyche PRNG.
    """
    def __init__(self, block_size: int = 4, num_rounds: int = 4, backend:str = "jax"):
        
        if backend == "pallas":
            from tyche.backend_pallas import PallasBackend
            self._backend = PallasBackend(num_rounds, block_size)
        else:
            from tyche.backend_jax import JaxBackend
            self._backend = JaxBackend(num_rounds, block_size)
        
        if block_size not in (4, 8, 16, 32):
            raise ValueError(f"block_size must be one of 4, 8, 16, 32 — got {block_size}")
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1 — got {num_rounds}")

        self.block_size = block_size
        self.num_rounds = num_rounds
        self._hash_parallel = make_hash_parallel(num_rounds)

    @property
    def key_shape(self):
        return (key_size_uint32(self.num_rounds, self.block_size),)

    @property
    def name(self):
        return f"tyche_b{self.block_size}_r{self.num_rounds}"

    def _seed(self, seed) -> jnp.ndarray:
        return expand_seed_to_key(seed, self.num_rounds, self.block_size)

    def _split(self, key: jnp.ndarray, shape: tuple) -> jnp.ndarray:
        num = math.prod(shape)
        child_indices = jnp.arange(num, dtype=jnp.uint64)
        children = jax.vmap(
            lambda i: derive_child_key(key, i, self.num_rounds, self.block_size)
        )(child_indices)
        return children.reshape(*shape, *self.key_shape)

    def _fold_in(self, key: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        return derive_child_key(
            key, data.astype(jnp.uint64), self.num_rounds, self.block_size
        )

    def _random_bits(self, key: jnp.ndarray, bit_width: int, shape: tuple) -> jnp.ndarray:
        
        """
        Generate random bits by hashing counter blocks with key matrices.
        This method calculates how many (B, B) blocks of output are needed to produce the requested number of bits,
        generates the necessary counter blocks, and then hashes them in parallel using the backend's hash_parallel
        method. Finally, it packs the resulting uint16 outputs into the requested bit width (32 or 64).
        """
        
        B, R = self.block_size, self.num_rounds

        total_out_elems = math.prod(shape)
        
        uint16_per_elem = bit_width // 16
        total_uint16 = total_out_elems * uint16_per_elem
        num_blocks = math.ceil(total_uint16 / (B * B))

        weight_matrices = _key_to_matrices(key, R, B)
        counter_blocks = make_counter_blocks(key, 0, num_blocks, B)
        hashed = self._hash_parallel(counter_blocks, weight_matrices)

        flat_u16 = hashed.reshape(-1)[:total_uint16]
        out_dtype = jnp.uint32 if bit_width == 32 else jnp.uint64
        packed = flat_u16.view(out_dtype)

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