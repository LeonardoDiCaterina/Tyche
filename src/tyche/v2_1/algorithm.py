import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Odd multiplier for ALU nonlinearity — bijection on Z_{2^32}
_ODD_MULT = jnp.uint32(0x94D049BB)

# SplitMix64 constants for seed expansion
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

def key_size_uint32(num_rounds: int, tile_size: int) -> int:
    """Number of uint32 words in a Tyche V2 key. 1 word per matrix element, so R * T * T."""
    return num_rounds * tile_size * tile_size

def _key_to_matrices(key: jnp.ndarray, num_rounds: int, tile_size: int) -> jnp.ndarray:
    """Unpack flat uint32 key directly to (NUM_ROUNDS, T, T) uint32 (which is treated as int32)."""
    return key.reshape(num_rounds, tile_size, tile_size)

def _matrices_to_key(matrices: jnp.ndarray) -> jnp.ndarray:
    """Pack matrices back to flat uint32 key."""
    return matrices.reshape(-1)

def expand_seed_to_key(seed, num_rounds: int, tile_size: int) -> jnp.ndarray:
    n_u32 = key_size_uint32(num_rounds, tile_size)
    n_u64 = (n_u32 + 1) // 2
    seed_u64 = jnp.array(seed, dtype=jnp.uint64)
    _, words_u64 = jax.lax.scan(
        lambda s, _: _splitmix64_step(s),
        seed_u64, None, length=n_u64
    )
    return _u64_to_u32_array(words_u64, n_u32)

def _mix_key_const(key: jnp.ndarray) -> jnp.uint32: # type: ignore
    folded = jnp.bitwise_xor.reduce(key.astype(jnp.uint32))
    return folded * jnp.uint32(2654435761)

# Fast 2-multiply bijective hash
_FAST_MUL1 = jnp.uint32(0xBF58476D)
_FAST_MUL2 = jnp.uint32(0x94D049BB)

def _fast_mix_u32(x: jnp.ndarray) -> jnp.ndarray:
    x = (x ^ (x >> jnp.uint32(16))) * _FAST_MUL1
    x = (x ^ (x >> jnp.uint32(13))) * _FAST_MUL2
    x = x ^ (x >> jnp.uint32(16))
    return x

def make_tiles(key: jnp.ndarray, offset: int, num_tiles: int, tile_size: int, embedding: str) -> jnp.ndarray:
    T = tile_size
    key_mix = _mix_key_const(key).astype(jnp.uint32)
    n = (jnp.arange(num_tiles, dtype=jnp.uint32) + jnp.uint32(offset)).reshape((num_tiles, 1, 1))
    
    rows = jnp.arange(T, dtype=jnp.uint32)
    cols = jnp.arange(T, dtype=jnp.uint32)
    R, C = jnp.meshgrid(rows, cols, indexing='ij')
    R = R.reshape((1, T, T))
    C = C.reshape((1, T, T))

    M1 = jnp.uint32(2654435761)
    M2 = jnp.uint32(1234567891)
    M3 = jnp.uint32(987654321)

    if embedding == "hash":
        raw = key_mix ^ (n * M1) ^ (R * M2) ^ (C * M3)
        mixed = _fast_mix_u32(raw)
        return mixed.astype(jnp.int8)
    
    elif embedding == "diagonal":
        mixed = _fast_mix_u32(key_mix ^ n)
        return jnp.where(R == C, mixed.astype(jnp.int8), jnp.int8(0))
        
    elif embedding == "row":
        raw = key_mix ^ n ^ (R * M2)
        mixed = _fast_mix_u32(raw)
        return mixed.astype(jnp.int8)
        
    elif embedding == "rank1":
        v1 = _fast_mix_u32(key_mix ^ n ^ R)
        v2 = _fast_mix_u32(key_mix ^ n ^ C)
        return (v1 * v2).astype(jnp.int8)
        
    else:
        raise ValueError(f"Unknown embedding mode: {embedding}")

def _hash_tile(tile_int8: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
    """
    JAX Fallback loop for V2.1: INT8 -> INT32 -> INT8.
    """
    def round_fn(x, W_r):
        x_int32 = x.astype(jnp.int32)
        W_int32 = W_r.astype(jnp.int32)
        
        acc_32 = jnp.matmul(x_int32, x_int32) + W_int32
        
        # ODD MULT is uint32, so cast acc to uint32 for the bitwise ops
        acc_u32 = acc_32.view(jnp.uint32)
        acc_u32 = acc_u32 * _ODD_MULT
        alu_mixed = acc_u32 ^ (acc_u32 >> jnp.uint32(16))
        
        # Cast back to int8
        return alu_mixed.astype(jnp.int8), None

    x, _ = jax.lax.scan(round_fn, tile_int8, weight_matrices)
    return x

def make_hash_parallel(num_rounds: int):
    @jax.jit
    def hash_parallel(tiles: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(_hash_tile, in_axes=(0, None))(tiles, weight_matrices)
    return hash_parallel

def _expand_scalar_to_matrix(value: jnp.ndarray, tile_size: int) -> jnp.ndarray:
    T = tile_size
    n = T * T
    base = value.astype(jnp.uint32)
    indices = jnp.arange(n, dtype=jnp.uint32)
    raw = base + indices * jnp.uint32(0x9E3779B9)
    return _fast_mix_u32(raw).reshape(T, T)

def _apply_perturbation(weight_matrices: jnp.ndarray, perturbation: jnp.ndarray) -> jnp.ndarray:
    def perturb_round(W_r):
        x = jnp.matmul(W_r.astype(jnp.uint32), W_r.astype(jnp.uint32)) + perturbation.astype(jnp.uint32)
        return x
    return jax.vmap(perturb_round)(weight_matrices)

def derive_child_key(key: jnp.ndarray, value: jnp.ndarray, num_rounds: int, tile_size: int) -> jnp.ndarray:
    weight_matrices = _key_to_matrices(key, num_rounds, tile_size)
    round_indices = jnp.arange(num_rounds, dtype=jnp.uint32)
    hashed_value = _fast_mix_u32(value.astype(jnp.uint32))

    def perturb_round(W_r, r):
        round_salt = r * jnp.uint32(0x9E3779B9)
        P = _expand_scalar_to_matrix(hashed_value ^ round_salt, tile_size)
        return jnp.matmul(W_r.astype(jnp.uint32), W_r.astype(jnp.uint32)) + P

    new_matrices = jax.vmap(perturb_round)(weight_matrices, round_indices)
    return _matrices_to_key(new_matrices)
