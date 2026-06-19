import jax
import jax.numpy as jnp

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

def key_size_uint(num_rounds: int, tile_size: int, word_size: int) -> int:
    # Always return size in terms of uint32 elements to satisfy JAX PRNG interface
    base_words = num_rounds * tile_size * tile_size
    return base_words * (word_size // 32)

def _key_to_matrices(key: jnp.ndarray, num_rounds: int, tile_size: int, word_size: int) -> jnp.ndarray:
    if word_size == 64:
        key = key.view(jnp.uint64)
    return key.reshape(num_rounds, tile_size, tile_size)

def _matrices_to_key(matrices: jnp.ndarray) -> jnp.ndarray:
    return matrices.view(jnp.uint32).reshape(-1)

def expand_seed_to_key(seed, num_rounds: int, tile_size: int, word_size: int) -> jnp.ndarray:
    n_u32 = key_size_uint(num_rounds, tile_size, word_size)
    n_u64 = (n_u32 + 1) // 2
    seed_u64 = jnp.array(seed, dtype=jnp.uint64)
    _, words_u64 = jax.lax.scan(lambda s, _: _splitmix64_step(s), seed_u64, None, length=n_u64)
    return _u64_to_u32_array(words_u64, n_u32)

def _mix_key_const(key: jnp.ndarray, word_size: int):
    if word_size == 32:
        folded = jnp.bitwise_xor.reduce(key.astype(jnp.uint32))
        return folded * jnp.uint32(2654435761)
    else:
        folded = jnp.bitwise_xor.reduce(key.astype(jnp.uint64))
        return folded * jnp.uint64(0x9E3779B97F4A7C15)

_FAST_MUL1_32 = jnp.uint32(0xBF58476D)
_FAST_MUL2_32 = jnp.uint32(0x94D049BB)
_FAST_MUL1_64 = jnp.uint64(0xBF58476D1CE4E5B9)
_FAST_MUL2_64 = jnp.uint64(0x94D049BB133111EB)

def _fast_mix(x: jnp.ndarray, word_size: int) -> jnp.ndarray:
    if word_size == 32:
        x = (x ^ (x >> jnp.uint32(16))) * _FAST_MUL1_32
        x = (x ^ (x >> jnp.uint32(13))) * _FAST_MUL2_32
        x = x ^ (x >> jnp.uint32(16))
    else:
        x = (x ^ (x >> jnp.uint64(30))) * _FAST_MUL1_64
        x = (x ^ (x >> jnp.uint64(27))) * _FAST_MUL2_64
        x = x ^ (x >> jnp.uint64(31))
    return x

def make_tiles(key: jnp.ndarray, offset: int, num_tiles: int, tile_size: int, embedding: str, word_size: int) -> jnp.ndarray:
    T = tile_size
    dtype = jnp.uint32 if word_size == 32 else jnp.uint64
    key_mix = _mix_key_const(key, word_size).astype(dtype)
    n = (jnp.arange(num_tiles, dtype=dtype) + dtype(offset)).reshape((num_tiles, 1, 1))
    
    rows = jnp.arange(T, dtype=dtype)
    cols = jnp.arange(T, dtype=dtype)
    R, C = jnp.meshgrid(rows, cols, indexing='ij')
    R = R.reshape((1, T, T))
    C = C.reshape((1, T, T))

    M1 = dtype(2654435761)
    M2 = dtype(1234567891)
    M3 = dtype(987654321)

    if embedding == "hash":
        raw = key_mix ^ (n * M1) ^ (R * M2) ^ (C * M3)
        mixed = _fast_mix(raw, word_size)
        return mixed.astype(dtype)
    elif embedding == "diagonal":
        mixed = _fast_mix(key_mix ^ n, word_size)
        return jnp.where(R == C, mixed.astype(dtype), dtype(0))
    elif embedding == "row":
        raw = key_mix ^ n ^ (R * M2)
        mixed = _fast_mix(raw, word_size)
        return mixed.astype(dtype)
    elif embedding == "rank1":
        v1 = _fast_mix(key_mix ^ n ^ R, word_size)
        v2 = _fast_mix(key_mix ^ n ^ C, word_size)
        return (v1 * v2).astype(dtype)
    else:
        raise ValueError(f"Unknown embedding mode: {embedding}")

def _hash_tile(tile: jnp.ndarray, weight_matrices: jnp.ndarray, word_size: int) -> jnp.ndarray:
    # Philox round logic (using standard JAX ops as fallback)
    T = tile.shape[0]
    total_elements = T * T
    flat_tile = tile.reshape(-1)
    
    if word_size == 32:
        # Philox-4x32
        M0 = jnp.uint32(0xCD9E8D57)
        M1 = jnp.uint32(0xD2511F53)
        
        def round_fn(x_flat, W_r):
            # x_flat is flat, W_r is T x T matrix
            # We treat W_r as two keys k0, k1 per chunk of 4
            W_flat = W_r.reshape(-1)
            chunk_size = total_elements // 4
            
            x0 = x_flat[0:chunk_size]
            x1 = x_flat[chunk_size:2*chunk_size]
            x2 = x_flat[2*chunk_size:3*chunk_size]
            x3 = x_flat[3*chunk_size:4*chunk_size]
            
            k0 = W_flat[0:chunk_size]
            k1 = W_flat[chunk_size:2*chunk_size]
            
            p0 = x0.astype(jnp.uint64) * M0.astype(jnp.uint64)
            hi0 = (p0 >> 32).astype(jnp.uint32)
            lo0 = (p0 & 0xFFFFFFFF).astype(jnp.uint32)
            
            p1 = x2.astype(jnp.uint64) * M1.astype(jnp.uint64)
            hi1 = (p1 >> 32).astype(jnp.uint32)
            lo1 = (p1 & 0xFFFFFFFF).astype(jnp.uint32)
            
            nx0 = hi1 ^ k0 ^ x3
            nx1 = lo1
            nx2 = hi0 ^ k1 ^ x1
            nx3 = lo0
            
            new_x = jnp.concatenate([nx0, nx1, nx2, nx3])
            return new_x, None
    else:
        # Philox-2x64
        M0 = jnp.uint64(0xD2B74407B1CE6E93)
        
        def mulhi64(a, b):
            a_lo = (a & 0xFFFFFFFF).astype(jnp.uint64)
            a_hi = (a >> 32).astype(jnp.uint64)
            b_lo = (b & 0xFFFFFFFF).astype(jnp.uint64)
            b_hi = (b >> 32).astype(jnp.uint64)
            lo_lo = a_lo * b_lo
            hi_lo = a_hi * b_lo
            lo_hi = a_lo * b_hi
            hi_hi = a_hi * b_hi
            cross = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFF) + (lo_hi & 0xFFFFFFFF)
            upper = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (cross >> 32)
            return upper.astype(jnp.uint64)

        def round_fn(x_flat, W_r):
            W_flat = W_r.reshape(-1)
            chunk_size = total_elements // 2
            
            x0 = x_flat[0:chunk_size]
            x1 = x_flat[chunk_size:2*chunk_size]
            
            k0 = W_flat[0:chunk_size]
            
            lo0 = x0 * M0
            hi0 = mulhi64(x0, M0)
            
            nx0 = hi0 ^ k0 ^ x1
            nx1 = lo0
            
            new_x = jnp.concatenate([nx0, nx1])
            return new_x, None

    x, _ = jax.lax.scan(round_fn, flat_tile, weight_matrices)
    return x.reshape((T, T))

def make_hash_parallel(num_rounds: int, word_size: int):
    @jax.jit
    def hash_parallel(tiles: jnp.ndarray, weight_matrices: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda t, w: _hash_tile(t, w, word_size), in_axes=(0, None))(tiles, weight_matrices)
    return hash_parallel

def derive_child_key(key: jnp.ndarray, value: jnp.ndarray, num_rounds: int, tile_size: int, word_size: int) -> jnp.ndarray:
    # Simpler linear mixer to derive child keys (Ablation to eliminate matmul overhead)
    dtype = jnp.uint32 if word_size == 32 else jnp.uint64
    salt = _fast_mix(value.astype(dtype), word_size)
    key_flat = key.reshape(-1)
    indices = jnp.arange(key_flat.shape[0], dtype=dtype)
    new_key = _fast_mix(key_flat ^ salt ^ indices, word_size)
    return new_key.reshape(-1)
