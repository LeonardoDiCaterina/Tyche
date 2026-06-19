import jax
import jax.numpy as jnp

# We reuse the same key expansion as Philox to ensure apples-to-apples comparison of the generator loop.
from tyche.v3_philox.algorithm import (
    key_size_uint,
    expand_seed_to_key,
    _key_to_matrices,
    _matrices_to_key,
    _mix_key_const,
    _fast_mix,
    derive_child_key,
    make_tiles
)

def _hash_tile(tile: jnp.ndarray, weight_matrices: jnp.ndarray, word_size: int) -> jnp.ndarray:
    # JAX fallback for Threefry-like SP network over T x T tile.
    T = tile.shape[0]
    total_elements = T * T
    flat_tile = tile.reshape(-1)
    
    if word_size == 32:
        # Rotation constants for Threefry-4x32
        R_CONSTS = jnp.array([
            [10, 26], [11, 21], [13, 27], [23, 5],
            [6, 20], [17, 11], [25, 10], [18, 20]
        ], dtype=jnp.uint32)
        
        def rotl(x, k):
            return (x << k) | (x >> (32 - k))
            
        def round_fn(x_flat_and_r, W_r):
            x_flat, r_idx = x_flat_and_r
            W_flat = W_r.reshape(-1)
            
            # Key injection
            x_flat = x_flat + W_flat
            
            chunk_size = total_elements // 4
            x0 = x_flat[0:chunk_size]
            x1 = x_flat[chunk_size:2*chunk_size]
            x2 = x_flat[2*chunk_size:3*chunk_size]
            x3 = x_flat[3*chunk_size:4*chunk_size]
            
            rot1 = R_CONSTS[r_idx % 8, 0]
            rot2 = R_CONSTS[r_idx % 8, 1]
            
            # Mix
            x0 = x0 + x1
            x1 = rotl(x1, rot1)
            x1 = x1 ^ x0
            
            x2 = x2 + x3
            x3 = rotl(x3, rot2)
            x3 = x3 ^ x2
            
            # Permute: 0, 3, 1, 2
            new_x = jnp.concatenate([x0, x3, x1, x2])
            return (new_x, r_idx + 1), None
            
    else:
        # Rotation constants for Threefry-2x64
        R_CONSTS = jnp.array([16, 42, 12, 31, 16, 32, 24, 21], dtype=jnp.uint64)
        
        def rotl(x, k):
            return (x << k) | (x >> (64 - k))
            
        def round_fn(x_flat_and_r, W_r):
            x_flat, r_idx = x_flat_and_r
            W_flat = W_r.reshape(-1)
            
            x_flat = x_flat + W_flat
            
            chunk_size = total_elements // 2
            x0 = x_flat[0:chunk_size]
            x1 = x_flat[chunk_size:2*chunk_size]
            
            rot = R_CONSTS[r_idx % 8]
            
            x0 = x0 + x1
            x1 = rotl(x1, rot)
            x1 = x1 ^ x0
            
            new_x = jnp.concatenate([x0, x1])
            return (new_x, r_idx + 1), None

    (x, _), _ = jax.lax.scan(round_fn, (flat_tile, jnp.uint32(0)), weight_matrices)
    return x.reshape((T, T))

def make_hash_parallel(num_rounds: int, word_size: int):
    @jax.jit
    def hash_parallel(tiles: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda t, w: _hash_tile(t, w, word_size), in_axes=(0, None))(tiles, weight_matrices)
    return hash_parallel
