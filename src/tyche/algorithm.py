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
      x = x * ODD_MULT                Odd-multiply bijection (full carry cascade)
      x = x ^ (x >> 16)               XOR fold (high-bit entropy → low bits)
      x = cast(x, uint16)             Truncate back to Z_{2^16}
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

    We apply the odd/even masks in-place; this keeps all 16 input bits
    except the constrained LSBs and avoids throwing away entropy.
    
    Guarantees invertibility mod 2^k by construction:
    - Diagonal entries forced ODD (set LSB) → units in Z_65536
    - Upper triangle entries forced EVEN (clear LSB)
    - Lower triangle entries left unchanged
    
    This triangular map with odd diagonal is invertible by a standard
    lifting argument over Z_{2^k}.
    """
    B = block_size
    matrix = raw_data_16[:B * B].reshape((B, B)).astype(jnp.uint16)

    r, c = jnp.indices((B, B))
    matrix = jnp.where(r < c, matrix & jnp.uint16(0xFFFE), matrix)
    matrix = jnp.where(r == c, matrix | jnp.uint16(1), matrix)
    return matrix

# Odd multiplier for ALU nonlinearity — bijection on Z_{2^32}
# (from SplitMix64's finaliser; odd -> invertible mod 2^32)
_ODD_MULT = jnp.uint32(0x94D049BB)

# SplitMix64 constants for seed expansion
_SM64_ADD  = jnp.uint64(0x9E3779B97F4A7C15)
_SM64_MIX1 = jnp.uint64(0xBF58476D1CE4E5B9)
_SM64_MIX2 = jnp.uint64(0x94D049BB133111EB)

def _splitmix64_step(state: jnp.ndarray):
    """
    Perform one step of SplitMix64
    to generate the next uint64 word from the state.
    """
    state = (state + _SM64_ADD).astype(jnp.uint64)
    z = state
    z = ((z ^ (z >> jnp.uint64(30))) * _SM64_MIX1).astype(jnp.uint64)
    z = ((z ^ (z >> jnp.uint64(27))) * _SM64_MIX2).astype(jnp.uint64)
    return state, (z ^ (z >> jnp.uint64(31))).astype(jnp.uint64)

def _u64_to_u32_array(words_u64: jnp.ndarray, n_u32: int) -> jnp.ndarray:
    """Convert an array of uint64 words
    to a flat array of uint32, 
    taking the lower 32 bits first.
    """
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
        """
        In each round we compute x² + W_r with 32-bit accumulation,
        then apply an odd multiply bijection to break low-bit linearity,
        followed by an XOR fold to mix high-bit entropy down.
        
        The path is 
        16→32-bit MAC + 32-bit key addition
        → 32-bit odd multiply bijection
        → 32-bit XOR fold
        → 16-bit truncation
        """
        x_u32 = x.astype(jnp.uint32)
        acc_32 = jnp.matmul(x_u32, x_u32) + W_r
        acc_32 = acc_32 * _ODD_MULT        
        alu_mixed = acc_32 ^ (acc_32 >> jnp.uint32(16))
        
        return alu_mixed.astype(jnp.uint16), None

    x, _ = jax.lax.scan(round_fn, counter_block, weight_matrices)
    return x

def make_hash_parallel(num_rounds: int):
    @jax.jit
    def hash_parallel(counter_blocks: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(_hash_block, in_axes=(0, None))(counter_blocks, weight_matrices)
    return hash_parallel

def _mix_key_const(key: jnp.ndarray) -> jnp.uint32: # type: ignore
    """Combine all words of the key into a single 32-bit mixing constant.

    Previously only key[0] was used, leaving the remainder of a large key
    completely ignored by counter block generation.  We now xor-fold the
    entire key and apply a golden-ratio offset to spread entropy.
    
    xor-reduce then multiply by 2654435761 (Knuth's golden ratio)
    gives a simple bijective hash with good avalanche for small input changes.
    This ensures that all key bits influence the counter block generation,
    and that similar keys produce very different counter blocks
    """
    folded = jnp.bitwise_xor.reduce(key.astype(jnp.uint32))
    return folded * jnp.uint32(2654435761)


def make_counter_blocks(key: jnp.ndarray, offset: int, num_blocks: int, block_size: int) -> jnp.ndarray:
    B = block_size
    key_mix = _mix_key_const(key)
    block_indices = jnp.arange(num_blocks, dtype=jnp.uint32) + jnp.uint32(offset)

    def make_block(idx):
        """
        Generate one (B, B) uint16 counter block from key_mix and block index.
         - We combine key_mix and block index with element indices to get a unique uint32 for each element
         - Then we apply a bijective hash to get well-distributed uint32 values
         - Finally we embed to GL_B(Z_65536) with the same odd/even masking as the key embedding,
            ensuring invertibility and good diffusion.  
        
        This design ensures that each block is uniquely determined by the key and block index,
        and that similar keys or indices produce very different counter blocks.
        """
    
        rows = jnp.arange(B, dtype=jnp.uint32)
        cols = jnp.arange(B, dtype=jnp.uint32)
        R, C = jnp.meshgrid(rows, cols, indexing='ij')
        v = key_mix ^ (idx  * jnp.uint32(2654435761))
        v = v        ^ (R   * jnp.uint32(1234567891))
        v = v        ^ (C   * jnp.uint32(987654321))
        v = (v * jnp.uint32(1103515245) + jnp.uint32(12345)) ^ (v >> jnp.uint32(16))
        
        raw_16 = v.astype(jnp.uint16).reshape(-1)
        return tyche_embed(raw_16, B)

    return jax.vmap(make_block)(block_indices)

# -- Reduced 2-multiply bijective hash for perturbation expansion --------
# Fast, GPU-friendly alternative to full SplitMix64 scan.
# Two odd multiplies + XOR folds give strong avalanche for sequential IDs
# while staying branch-free and trivially lowerable to Pallas / thread-ID use.
_FAST_MUL1 = jnp.uint32(0xBF58476D)   # odd
_FAST_MUL2 = jnp.uint32(0x94D049BB)   # odd

def _fast_mix_u32(x: jnp.ndarray) -> jnp.ndarray:
    """2-multiply bijective hash: uint32 → uint32.  Branch-free, Pallas-ready."""
    x = (x ^ (x >> jnp.uint32(16))) * _FAST_MUL1
    x = (x ^ (x >> jnp.uint32(13))) * _FAST_MUL2
    x = x ^ (x >> jnp.uint32(16))
    return x

def _expand_scalar_to_matrix(value: jnp.ndarray, block_size: int) -> jnp.ndarray:
    """Expand a scalar (child index / fold-in data) into a (B, B) uint32 perturbation matrix.
    
    Uses a fast 2-multiply bijective hash seeded by value and element index.
    Designed so that in a Pallas kernel the value can be the thread ID directly.
    
    This gives a unique, well-diffused perturbation matrix for each child key or fold-in data,
    while being much faster than a full SplitMix64 expansion and still fully branch-free.
    """
    B = block_size
    n = B * B
    # Mix value with element indices to produce n independent-looking uint32s
    base = value.astype(jnp.uint32)
    indices = jnp.arange(n, dtype=jnp.uint32)
    # Combine base and index — golden-ratio offset avoids collisions for sequential values
    raw = base + indices * jnp.uint32(0x9E3779B9)
    return _fast_mix_u32(raw).reshape(B, B)

def _apply_perturbation(weight_matrices: jnp.ndarray, perturbation: jnp.ndarray) -> jnp.ndarray:
    def perturb_round(W_r):
        # Derive new keys via 32-bit quadratic map
        x = jnp.matmul(W_r, W_r) + perturbation
        return x
    return jax.vmap(perturb_round)(weight_matrices)

def derive_child_key(key: jnp.ndarray, value: jnp.ndarray, num_rounds: int, block_size: int) -> jnp.ndarray:
    
    """
    Derive a child key by perturbing the parent's weight matrices with a value-derived matrix.
    We generate a distinct perturbation for each round by incorporating the round index into the hash.
    This avoids siblings having identical offsets across all rounds, which would make them easy to correlate.
    """
    
    weight_matrices = _key_to_matrices(key, num_rounds, block_size)

    round_indices = jnp.arange(num_rounds, dtype=jnp.uint64)
    def perturb_round(W_r, r):
        """
        Perturb one round's weight matrix with a value-derived matrix.
        The perturbation is derived from the value and round index,
        ensuring unique perturbations across rounds and sibling keys.
        """
        P = _expand_scalar_to_matrix(value + r, block_size)
        return jnp.matmul(W_r, W_r) + P

    new_matrices = jax.vmap(perturb_round)(weight_matrices, round_indices)
    return _matrices_to_key(new_matrices)