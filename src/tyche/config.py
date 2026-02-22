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
    # Changed default num_rounds to 4 based on SAC tests
    def __init__(self, block_size: int = 4, num_rounds: int = 4):
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
        B, R = self.block_size, self.num_rounds

        total_out_elems = math.prod(shape)
        
        # Calculate how many 16-bit elements we need
        uint16_per_elem = bit_width // 16
        total_uint16 = total_out_elems * uint16_per_elem

        # Calculate blocks needed
        num_blocks = math.ceil(total_uint16 / (B * B))

        weight_matrices = _key_to_matrices(key, R, B)
        counter_blocks = make_counter_blocks(key, 0, num_blocks, B)
        
        # Hash outputs shape (num_blocks, B, B) of uint16
        hashed = self._hash_parallel(counter_blocks, weight_matrices)

        # Flatten and truncate to exact number of uint16s needed
        flat_u16 = hashed.reshape(-1)[:total_uint16]

        # Use JAX's zero-copy memory views to pack the 16-bit blocks into 32 or 64 bit outputs
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