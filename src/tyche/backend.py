from typing import Protocol
import jax.numpy as jnp


class TycheBackend(Protocol):
    """Swappable compute backend: pure-JAX (default) or Pallas kernel."""

    def hash_block(
        self,
        counter_block: jnp.ndarray,   # (B, B) uint16
        weight_matrices: jnp.ndarray,  # (R, B, B) uint32
    ) -> jnp.ndarray:                  # (B, B) uint16
        """Single-block hash: R rounds of X²+W with ALU fold."""
        ...

    def hash_parallel(
        self,
        counter_blocks: jnp.ndarray,   # (N, B, B) uint16
        weight_matrices: jnp.ndarray,   # (R, B, B) uint32
    ) -> jnp.ndarray:                   # (N, B, B) uint16
        """Batched hash over N counter blocks."""
        ...

    def apply_perturbation(
        self,
        weight_matrices: jnp.ndarray,  # (R, B, B) uint32
        perturbation: jnp.ndarray,     # (B, B) uint32
    ) -> jnp.ndarray:                  # (R, B, B) uint32
        """Derive new weight matrices: W_r² + P for each round."""
        ...

    def make_counter_blocks(
        self,
        key: jnp.ndarray,     # flat uint32 key
        offset: int,
        num_blocks: int,
        block_size: int,
    ) -> jnp.ndarray:         # (num_blocks, B, B) uint16
        """Build embedded counter matrices."""
        ...