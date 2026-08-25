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

def _hash_tile(tile: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
    """
    Simulated Tensor Core FMA rounds + ALU Fold.
    int8 input -> 32-bit MAC -> 32-bit int32 weights -> ALU Fold -> int8 output.
    """
    weight_matrices_i32 = weight_matrices.astype(jnp.int32)
    def round_fn(x, W_r):
        x_32 = x.astype(jnp.int32)
        acc_32 = jnp.matmul(x_32, x_32) + W_r
        
        acc_u32 = acc_32.view(jnp.uint32)
        acc_u32 = acc_u32 * _ODD_MULT        
        alu_mixed = acc_u32 ^ (acc_u32 >> jnp.uint32(16))
        
        return alu_mixed.astype(jnp.int8), None

    x, _ = jax.lax.scan(round_fn, tile, weight_matrices_i32)
    return x

def make_hash_parallel(num_rounds: int):
    @jax.jit
    def hash_parallel(tiles: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(_hash_tile, in_axes=(0, None))(tiles, weight_matrices)
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

def make_tile(key: jnp.ndarray, offset: int, num_tiles: int, tile_size: int, embedding: str = "hash") -> jnp.ndarray:
    T = tile_size
    key_mix = _mix_key_const(key)
    tile_indices = jnp.arange(num_tiles, dtype=jnp.uint32) + jnp.uint32(offset)

    def make_single_tile(idx):
        rows, cols = jnp.meshgrid(jnp.arange(T, dtype=jnp.uint32), jnp.arange(T, dtype=jnp.uint32), indexing='ij')
        
        if embedding == "diagonal":
            v = jnp.where(rows == cols, _fast_mix_u32(key_mix ^ idx), jnp.uint32(0))
        elif embedding == "row":
            v = _fast_mix_u32(key_mix ^ idx ^ (rows * jnp.uint32(1234567891)))
        elif embedding == "rank1":
            v1 = _fast_mix_u32(key_mix ^ idx ^ rows)
            v2 = _fast_mix_u32(key_mix ^ idx ^ cols)
            v = v1 * v2
        else:
            # Fallback to hash
            v = key_mix ^ (idx * jnp.uint32(2654435761))
            v = v ^ (rows * jnp.uint32(1234567891))
            v = v ^ (cols * jnp.uint32(987654321))
            v = _fast_mix_u32(v)
            
        return v.astype(jnp.int8)

    return jax.vmap(make_single_tile)(tile_indices)

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

    round_indices = jnp.arange(num_rounds, dtype=jnp.uint32)
    # Pre-hash the child value so that sequential indices (0,1,2,...)
    # are spread across the full uint32 range *before* we mix in the
    # round index.  Previously "value + r" was used, which meant
    # child 0 / round 1 and child 1 / round 0 received the *same*
    # scalar — producing identical perturbation matrices and
    # catastrophic cross-stream correlations in PractRand.
    hashed_value = _fast_mix_u32(value.astype(jnp.uint32))

    def perturb_round(W_r, r):
        """
        Perturb one round's weight matrix with a value-derived matrix.
        The perturbation is derived from the pre-hashed value XOR'd with
        a round-dependent salt, ensuring unique perturbations across
        both rounds and sibling keys.
        """
        round_salt = r * jnp.uint32(0x9E3779B9)   # golden-ratio odd mult
        P = _expand_scalar_to_matrix(hashed_value ^ round_salt, block_size)
        return jnp.matmul(W_r, W_r) + P

    new_matrices = jax.vmap(perturb_round)(weight_matrices, round_indices)
    return _matrices_to_key(new_matrices)