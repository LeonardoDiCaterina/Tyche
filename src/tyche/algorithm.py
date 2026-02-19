"""
tyche/algorithm.py

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
  Fully tensor-core compatible.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


LCG_MULT = jnp.int32(1103515245)
TRUNC_SHIFT = jnp.int32(24)

def tyche_embed(raw_bytes, block_size):
    """
    Embed raw bytes into a guaranteed invertible GL_B(Z_256) matrix.

    Guarantees invertibility mod 2^k by construction:
      - Diagonal entries forced ODD (set LSB) → units in Z_256
      - Upper triangle entries forced EVEN (clear LSB)
      - Lower triangle entries left unchanged

    This triangular map with odd diagonal is invertible by a standard
    lifting argument over Z_{2^k}.
    """
    B = block_size
    matrix = raw_bytes[:B * B].reshape((B, B)).astype(jnp.int8)
    r, c = jnp.indices((B, B))
    # Upper triangle EVEN (clear LSB with 0xFE), Diagonal ODD (set LSB with 0x01)
    matrix = jnp.where(r < c, matrix & jnp.int8(-2), matrix)
    matrix = jnp.where(r == c, matrix | jnp.int8(1), matrix)
    return matrix


# SplitMix64 constants for seed/perturbation expansion
_SM64_ADD  = jnp.uint64(0x9E3779B97F4A7C15)
_SM64_MIX1 = jnp.uint64(0xBF58476D1CE4E5B9)
_SM64_MIX2 = jnp.uint64(0x94D049BB133111EB)


def _splitmix64_step(state: jnp.ndarray):
    """One SplitMix64 step. (state, _) → (new_state, output uint64)."""
    state = (state + _SM64_ADD).astype(jnp.uint64)
    z = state
    z = ((z ^ (z >> jnp.uint64(30))) * _SM64_MIX1).astype(jnp.uint64)
    z = ((z ^ (z >> jnp.uint64(27))) * _SM64_MIX2).astype(jnp.uint64)
    return state, (z ^ (z >> jnp.uint64(31))).astype(jnp.uint64)


def _u64_to_u32_array(words_u64: jnp.ndarray, n_u32: int) -> jnp.ndarray:
    """Split uint64 array into interleaved uint32 array, trimmed to n_u32."""
    lo = (words_u64 & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32)
    hi = (words_u64 >> jnp.uint64(32)).astype(jnp.uint32)
    return jnp.stack([lo, hi], axis=1).reshape(-1)[:n_u32]


def key_size_uint32(num_rounds: int, block_size: int) -> int:
    """Number of uint32 words in a Tyche key."""
    total_int8 = num_rounds * block_size * block_size
    assert total_int8 % 4 == 0, "Key byte count must be divisible by 4"
    return total_int8 // 4

def _key_to_matrices(key: jnp.ndarray, num_rounds: int, block_size: int) -> jnp.ndarray:
    """
    Unpack flat uint32 key → (NUM_ROUNDS, BLOCK_SIZE, BLOCK_SIZE) int8.
    Extracts 4 bytes from each uint32 word.
    """
    R, B = num_rounds, block_size
    b0 = (key & jnp.uint32(0xFF)).astype(jnp.int8)
    b1 = ((key >> jnp.uint32(8))  & jnp.uint32(0xFF)).astype(jnp.int8)
    b2 = ((key >> jnp.uint32(16)) & jnp.uint32(0xFF)).astype(jnp.int8)
    b3 = ((key >> jnp.uint32(24)) & jnp.uint32(0xFF)).astype(jnp.int8)
    flat_int8 = jnp.stack([b0, b1, b2, b3], axis=1).reshape(-1)
    return flat_int8.reshape(R, B, B)


def _matrices_to_key(matrices: jnp.ndarray) -> jnp.ndarray:
    """
    Pack (NUM_ROUNDS, BLOCK_SIZE, BLOCK_SIZE) int8 → flat uint32 key.
    """
    flat = matrices.reshape(-1).view(jnp.uint8).astype(jnp.uint32)
    n_words = len(flat) // 4
    b = flat.reshape(n_words, 4)
    return (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24))


def expand_seed_to_key(seed, num_rounds: int, block_size: int) -> jnp.ndarray:
    """
    uint64 seed → flat uint32 key array of shape (key_size_uint32(...),).
    Uses SplitMix64 to fill all words.
    """
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
    Hash one (B, B) int8 block through quadratic FMA rounds.

    Each round:
      x = matmul(x, x) + W_r      # Quadratic map: X² + C  (tensor core FMA)
      x = (x * LCG_MULT) >> 24    # Truncation: propagates high-bit entropy to LSBs
      x = cast(int8)               # Back to Z_256

    The quadratic recurrence X² + C introduces non-linearity absent from
    affine generators. The LCG truncation step solves the "upward carry"
    weakness in the LSBs described in the notebook SAC analysis.
    """
    def round_fn(x, W_r):
        x_i32 = x.astype(jnp.int32)
        w_i32 = W_r.astype(jnp.int32)
        x_new = jnp.matmul(x_i32, x_i32) + w_i32
        x_new = (x_new * LCG_MULT) >> TRUNC_SHIFT
        return x_new.astype(jnp.int8), None

    x, _ = jax.lax.scan(round_fn, counter_block, weight_matrices)
    return x

def make_hash_parallel(num_rounds: int):
    """Returns a jitted, vmapped hash function for the given num_rounds."""

    @jax.jit
    def hash_parallel(counter_blocks: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
        """
        Hash (NUM_BLOCKS, B, B) counter blocks in parallel.
        weight_matrices: (NUM_ROUNDS, B, B) int8 — shared across all blocks.
        Returns: (NUM_BLOCKS, B, B) int8
        """
        return jax.vmap(_hash_block, in_axes=(0, None))(counter_blocks, weight_matrices)

    return hash_parallel

def make_counter_blocks(
    key: jnp.ndarray,
    offset: int,
    num_blocks: int,
    block_size: int
) -> jnp.ndarray:
    """
    Build (num_blocks, block_size, block_size) int8 counter blocks.
    Each block is uniquely indexed by (key fingerprint, offset + block_idx).
    """
    B = block_size
    key_mix = key[0]  # use first uint32 word as mixing constant
    block_indices = jnp.arange(num_blocks, dtype=jnp.uint32) + jnp.uint32(offset)

    def make_block(idx):
        rows = jnp.arange(B, dtype=jnp.uint32)
        cols = jnp.arange(B, dtype=jnp.uint32)
        R, C = jnp.meshgrid(rows, cols, indexing='ij')
        v = key_mix ^ (idx  * jnp.uint32(2654435761))
        v = v        ^ (R   * jnp.uint32(1234567891))
        v = v        ^ (C   * jnp.uint32(987654321))
        v = (v * jnp.uint32(1103515245) + jnp.uint32(12345)) ^ (v >> jnp.uint32(16))
        # Embed into GL_B(Z_256) to guarantee invertibility
        raw_bytes = v.astype(jnp.uint8).reshape(-1)
        return tyche_embed(raw_bytes, B)

    return jax.vmap(make_block)(block_indices)


def _expand_scalar_to_matrix(value: jnp.ndarray, block_size: int) -> jnp.ndarray:
    """
    Expand a uint64 scalar into a (BLOCK_SIZE, BLOCK_SIZE) int8 perturbation matrix.
    Uses SplitMix64 to fill the matrix entries.
    """
    B = block_size
    n_u32 = (B * B + 3) // 4
    n_u64 = (n_u32 + 1) // 2

    _, words_u64 = jax.lax.scan(
        lambda s, _: _splitmix64_step(s),
        value.astype(jnp.uint64), None, length=n_u64
    )
    flat_u32 = _u64_to_u32_array(words_u64, n_u32)

    # Extract bytes → int8
    b0 = (flat_u32 & jnp.uint32(0xFF)).astype(jnp.int8)
    b1 = ((flat_u32 >> jnp.uint32(8))  & jnp.uint32(0xFF)).astype(jnp.int8)
    b2 = ((flat_u32 >> jnp.uint32(16)) & jnp.uint32(0xFF)).astype(jnp.int8)
    b3 = ((flat_u32 >> jnp.uint32(24)) & jnp.uint32(0xFF)).astype(jnp.int8)
    flat_int8 = jnp.stack([b0, b1, b2, b3], axis=1).reshape(-1)[:B * B]
    return flat_int8.reshape(B, B)


def _apply_perturbation(
    weight_matrices: jnp.ndarray,
    perturbation: jnp.ndarray,
) -> jnp.ndarray:
    """
    Derive new weight matrices via quadratic perturbation:
      W_r_new = truncate(W_r² + P)

    Uses the same quadratic map + LCG truncation as the core hash.
    """
    def perturb_round(W_r):
        x = jnp.matmul(W_r.astype(jnp.int32), W_r.astype(jnp.int32))
        x = (x + perturbation.astype(jnp.int32))
        x = (x * LCG_MULT) >> TRUNC_SHIFT
        return x.astype(jnp.int8)

    return jax.vmap(perturb_round)(weight_matrices)


def derive_child_key(
    key: jnp.ndarray,
    value: jnp.ndarray,
    num_rounds: int,
    block_size: int,
) -> jnp.ndarray:
    """
    Derive a child key from a parent key and a uint64 value.
    Used by both split (value = child index) and fold_in (value = data).

    Steps:
      1. Expand value → perturbation matrix P (B, B) int8
      2. W_r_child = W_r² + P  for all r
      3. Pack back to flat uint32 key
    """
    weight_matrices = _key_to_matrices(key, num_rounds, block_size)
    perturbation = _expand_scalar_to_matrix(value, block_size)
    new_matrices = _apply_perturbation(weight_matrices, perturbation)
    return _matrices_to_key(new_matrices)