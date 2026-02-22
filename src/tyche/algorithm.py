"""
Core Tyche PRNG algorithm — Quadratic Maps over GL_B(Z_256).

Design principles:
  - Stateless counter-mode (like Threefry/Philox)
  - Quadratic map X² + W as primary mixing function (tensor core compatible)
  - GL_B(Z_256) embedding guarantees invertibility via triangular structure
  - All key operations (split, fold_in) use quadratic perturbation

Key structure:
  Flat uint32 array storing (NUM_ROUNDS, BLOCK_SIZE, BLOCK_SIZE) int8 matrices.
  Each round has one weight matrix W_r (the additive constant in X² + W).

Algorithm per block:
  x = tyche_embed(counter_block)       Embed into GL_B(Z_256)
  for r in range(NUM_ROUNDS):
      x = matmul(x, x) + W_r          Quadratic FMA (int32 accumulation)
      x = (x * LCG_MULT) >> 24        Truncation non-linearity (keeps high 8 bits)
      x = cast(x, int8)               Back to Z_256
  return x

split / fold_in:
  Derive a perturbation matrix P from (child index / data),
  then W_r_new = W_r² + P for each round.
  Fully tensor-core compatible."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def tyche_embed(raw_data_16, block_size):
    """
    Embed raw 16-bit data into a guaranteed invertible GL_B(Z_65536) matrix.
    """
    B = block_size
    matrix = raw_data_16[:B * B].reshape((B, B)).astype(jnp.uint16)
    
    # SHIFT LEFT by 1 to protect the original entropy (Bit 0) from structural masks
    matrix = matrix << 1
    
    r, c = jnp.indices((B, B))
    # Upper triangle EVEN (clear LSB), Diagonal ODD (set LSB)
    matrix = jnp.where(r < c, matrix & jnp.uint16(0xFFFE), matrix)
    matrix = jnp.where(r == c, matrix | jnp.uint16(1), matrix)
    return matrix

# SplitMix64 constants for seed/perturbation expansion
_SM64_ADD  = jnp.uint64(0x9E3779B97F4A7C15)
_SM64_MIX1 = jnp.uint64(0xBF58476D1CE4E5B9)
_SM64_MIX2 = jnp.uint64(0x94D049BB133111EB)

def _splitmix64_step(state: jnp.ndarray):
    state = (state + _SM64_ADD).astype(jnp.uint64)
    z = state
    z = ((z ^ (z >> jnp.uint64(30))) * _SM64_MIX1).astype(jnp.uint64)
    z = ((z ^ (z >> jnp.uint64(27))) * _SM64_MIX2).astype(jnp.uint64)
    return state, (z ^ (z >> jnp.uint64(31))).astype(jnp.uint64)

def _u64_to_u32_array(words_u64: jnp.ndarray, n_u32: int) -> jnp.ndarray:
    lo = (words_u64 & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32)
    hi = (words_u64 >> jnp.uint64(32)).astype(jnp.uint32)
    return jnp.stack([lo, hi], axis=1).reshape(-1)[:n_u32]

def key_size_uint32(num_rounds: int, block_size: int) -> int:
    """Number of uint32 words in a Tyche key. Now exactly 1 word per matrix element."""
    return num_rounds * block_size * block_size

def _key_to_matrices(key: jnp.ndarray, num_rounds: int, block_size: int) -> jnp.ndarray:
    """Unpack flat uint32 key directly to (NUM_ROUNDS, BLOCK_SIZE, BLOCK_SIZE) uint32."""
    return key.reshape(num_rounds, block_size, block_size)

def _matrices_to_key(matrices: jnp.ndarray) -> jnp.ndarray:
    """Pack matrices back to flat uint32 key."""
    return matrices.reshape(-1)

def expand_seed_to_key(seed, num_rounds: int, block_size: int) -> jnp.ndarray:
    n_u32 = key_size_uint32(num_rounds, block_size)
    n_u64 = (n_u32 + 1) // 2
    seed_u64 = jnp.array(seed, dtype=jnp.uint64)
    _, words_u64 = jax.lax.scan(
        lambda s, _: _splitmix64_step(s),
        seed_u64, None, length=n_u64
    )
    return _u64_to_u32_array(words_u64, n_u32)

def _hash_block(counter_block: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
    """
    Simulated Tensor Core FMA rounds + ALU Fold.
    16-bit input -> 32-bit MAC -> 32-bit uint32 weights -> ALU Fold -> 16-bit output.
    """
    def round_fn(x, W_r):
        x_u32 = x.astype(jnp.uint32)
        
        # MAC (Accumulation natively in 32-bit, adding 32-bit entropy from W_r)
        acc_32 = jnp.matmul(x_u32, x_u32) + W_r
        
        # ALU Bridge: Fold high bits into low bits to capture full cross-multiplication
        alu_mixed = acc_32 ^ (acc_32 >> jnp.uint32(16))
        
        # Truncate back to uint16
        return alu_mixed.astype(jnp.uint16), None

    x, _ = jax.lax.scan(round_fn, counter_block, weight_matrices)
    return x

def make_hash_parallel(num_rounds: int):
    @jax.jit
    def hash_parallel(counter_blocks: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(_hash_block, in_axes=(0, None))(counter_blocks, weight_matrices)
    return hash_parallel

def make_counter_blocks(key: jnp.ndarray, offset: int, num_blocks: int, block_size: int) -> jnp.ndarray:
    B = block_size
    key_mix = key[0]  
    block_indices = jnp.arange(num_blocks, dtype=jnp.uint32) + jnp.uint32(offset)

    def make_block(idx):
        rows = jnp.arange(B, dtype=jnp.uint32)
        cols = jnp.arange(B, dtype=jnp.uint32)
        R, C = jnp.meshgrid(rows, cols, indexing='ij')
        v = key_mix ^ (idx  * jnp.uint32(2654435761))
        v = v        ^ (R   * jnp.uint32(1234567891))
        v = v        ^ (C   * jnp.uint32(987654321))
        v = (v * jnp.uint32(1103515245) + jnp.uint32(12345)) ^ (v >> jnp.uint32(16))
        
        # Cast to uint16 for the new embedding function
        raw_16 = v.astype(jnp.uint16).reshape(-1)
        return tyche_embed(raw_16, B)

    return jax.vmap(make_block)(block_indices)

def _expand_scalar_to_matrix(value: jnp.ndarray, block_size: int) -> jnp.ndarray:
    """Expands a uint64 scalar directly into a (B, B) uint32 perturbation matrix."""
    B = block_size
    n_u32 = B * B
    n_u64 = (n_u32 + 1) // 2

    _, words_u64 = jax.lax.scan(
        lambda s, _: _splitmix64_step(s),
        value.astype(jnp.uint64), None, length=n_u64
    )
    flat_u32 = _u64_to_u32_array(words_u64, n_u32)
    return flat_u32[:B * B].reshape(B, B)

def _apply_perturbation(weight_matrices: jnp.ndarray, perturbation: jnp.ndarray) -> jnp.ndarray:
    def perturb_round(W_r):
        # Derive new keys via 32-bit quadratic map
        x = jnp.matmul(W_r, W_r) + perturbation
        return x
    return jax.vmap(perturb_round)(weight_matrices)

def derive_child_key(key: jnp.ndarray, value: jnp.ndarray, num_rounds: int, block_size: int) -> jnp.ndarray:
    weight_matrices = _key_to_matrices(key, num_rounds, block_size)
    perturbation = _expand_scalar_to_matrix(value, block_size)
    new_matrices = _apply_perturbation(weight_matrices, perturbation)
    return _matrices_to_key(new_matrices)