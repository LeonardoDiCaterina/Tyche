"""
tyche/config.py

TycheConfig — configurable factory for the Tyche PRNGImpl.

Usage:
    from tyche.config import TycheConfig

    cfg = TycheConfig(block_size=4, num_rounds=16)
    impl = cfg.build()
    key = jax.random.PRNGKey(42, impl=impl)
"""

import math
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # required for uint64 arithmetic
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

    Uses quadratic maps over GL_B(Z_256) with Tensor Core-compatible
    FMA rounds: X_{n+1} = X_n² + W_n (mod 256).

    Args:
        block_size:  Square matrix dimension (default: 4 for GL_4(Z_256)).
                     Valid values: 4, 8, 16, 32.
        num_rounds:  Number of quadratic FMA rounds (default: 16).
                     More rounds = stronger diffusion, slightly slower.
    """

    def __init__(self, block_size: int = 4, num_rounds: int = 16):
        if block_size not in (4, 8, 16, 32):
            raise ValueError(f"block_size must be one of 4, 8, 16, 32 — got {block_size}")
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1 — got {num_rounds}")

        self.block_size = block_size
        self.num_rounds = num_rounds
        self._hash_parallel = make_hash_parallel(num_rounds)

    @property
    def key_shape(self):
        """Flat uint32 key — required by JAX's PRNGImpl."""
        return (key_size_uint32(self.num_rounds, self.block_size),)

    @property
    def name(self):
        return f"tyche_b{self.block_size}_r{self.num_rounds}"

    def _seed(self, seed) -> jnp.ndarray:
        """uint64 scalar → flat uint32 key."""
        return expand_seed_to_key(seed, self.num_rounds, self.block_size)

    def _split(self, key: jnp.ndarray, shape: tuple) -> jnp.ndarray:
        """key → (*shape, *key_shape) child keys, all derived via quadratic perturbation."""
        num = math.prod(shape)
        child_indices = jnp.arange(num, dtype=jnp.uint64)
        children = jax.vmap(
            lambda i: derive_child_key(key, i, self.num_rounds, self.block_size)
        )(child_indices)
        return children.reshape(*shape, *self.key_shape)

    def _fold_in(self, key: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        """Mix uint32 data into key via matmul perturbation."""
        return derive_child_key(
            key, data.astype(jnp.uint64), self.num_rounds, self.block_size
        )

    def _random_bits(self, key: jnp.ndarray, bit_width: int, shape: tuple) -> jnp.ndarray:
        """Generate random bits by hashing counter blocks with key matrices."""
        B, R = self.block_size, self.num_rounds

        total_out_elems = math.prod(shape)
        bytes_per_elem = bit_width // 8
        total_int8 = total_out_elems * bytes_per_elem

        num_blocks = math.ceil(total_int8 / (B * B))

        weight_matrices = _key_to_matrices(key, R, B)

        counter_blocks = make_counter_blocks(key, 0, num_blocks, B)
        hashed = self._hash_parallel(counter_blocks, weight_matrices)

        flat_bytes = hashed.reshape(-1)[:total_int8]
        flat_u8 = flat_bytes.view(jnp.uint8).reshape(-1, bytes_per_elem)

        out_dtype = jnp.uint32 if bit_width == 32 else jnp.uint64
        shift_dtype = jnp.uint32 if bit_width <= 32 else jnp.uint64
        work_dtype = jnp.uint32 if bit_width <= 32 else jnp.uint64
        shifts = jnp.array([8 * i for i in range(bytes_per_elem)], dtype=shift_dtype)
        packed = jnp.sum(flat_u8.astype(work_dtype) << shifts, axis=-1).astype(out_dtype)

        return packed.reshape(shape)

    def build(self) -> PRNGImpl:
        """Create and return the JAX PRNGImpl for this configuration."""
        return PRNGImpl(
            key_shape=self.key_shape,
            seed=self._seed,
            split=self._split,
            fold_in=self._fold_in,
            random_bits=self._random_bits,
            name=self.name,
            tag="tyc",
        )