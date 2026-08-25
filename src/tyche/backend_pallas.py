import jax.numpy as jnp
import jax
from jax.experimental import pallas as pl


class PallasBackend:
    def __init__(self, num_rounds: int, tile_size: int):
        self.R = num_rounds
        self.T = tile_size

    def hash_block(self, tile, weight_matrices):
        T = self.T

        def kernel(tile_ref, weights_ref, out_ref):
            x = tile_ref[...]
            for r in range(self.R):                         # unrolled at trace time
                W_r = weights_ref[r, :, :]                  # (T,T) uint32
                acc = pl.dot(x, x, out_dtype=jnp.int32) + W_r                    # fused matmul+add
                
                acc_u32 = acc.view(jnp.uint32)
                acc_u32 = acc_u32 * jnp.uint32(0x94D049BB)
                mixed = acc_u32 ^ (acc_u32 >> jnp.uint32(16))
                
                x = mixed.astype(jnp.int8)
            out_ref[...] = x

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((T, T), jnp.int8),
            grid=(1,),
            in_specs=[
                pl.BlockSpec((T, T), lambda i: (0, 0)),     # tile
                pl.BlockSpec((self.R, T, T), lambda i: (0, 0, 0)),  # weights
            ],
            out_specs=pl.BlockSpec((T, T), lambda i: (0, 0)),
        )(tile, weight_matrices.astype(jnp.int32))

    def hash_parallel(self, key, offset, num_tiles, weight_matrices, embedding):
        from tyche.algorithm import _mix_key_const
        N = num_tiles
        T = self.T
        key_mix = _mix_key_const(key)
        key_mix_arr = jnp.array(key_mix, dtype=jnp.uint32)

        def kernel(key_mix_ref, weights_ref, out_ref):
            i = jnp.uint32(pl.program_id(0) + offset)
            k_mix = key_mix_ref[...]
            
            rows = jnp.arange(T, dtype=jnp.uint32)[:, None]
            cols = jnp.arange(T, dtype=jnp.uint32)[None, :]
            
            if embedding == "diagonal":
                v_pre = k_mix ^ i
                v_mix = (v_pre ^ (v_pre >> jnp.uint32(16))) * jnp.uint32(0xBF58476D)
                v_mix = (v_mix ^ (v_mix >> jnp.uint32(13))) * jnp.uint32(0x94D049BB)
                v_mix = v_mix ^ (v_mix >> jnp.uint32(16))
                v = jnp.where(rows == cols, v_mix, jnp.uint32(0))
            elif embedding == "row":
                v_pre = k_mix ^ i ^ (rows * jnp.uint32(1234567891))
                v_mix = (v_pre ^ (v_pre >> jnp.uint32(16))) * jnp.uint32(0xBF58476D)
                v_mix = (v_mix ^ (v_mix >> jnp.uint32(13))) * jnp.uint32(0x94D049BB)
                v_mix = v_mix ^ (v_mix >> jnp.uint32(16))
                v = v_mix
            elif embedding == "rank1":
                v1_pre = k_mix ^ i ^ rows
                v1_mix = (v1_pre ^ (v1_pre >> jnp.uint32(16))) * jnp.uint32(0xBF58476D)
                v1_mix = (v1_mix ^ (v1_mix >> jnp.uint32(13))) * jnp.uint32(0x94D049BB)
                v1_mix = v1_mix ^ (v1_mix >> jnp.uint32(16))
                
                v2_pre = k_mix ^ i ^ cols
                v2_mix = (v2_pre ^ (v2_pre >> jnp.uint32(16))) * jnp.uint32(0xBF58476D)
                v2_mix = (v2_mix ^ (v2_mix >> jnp.uint32(13))) * jnp.uint32(0x94D049BB)
                v2_mix = v2_mix ^ (v2_mix >> jnp.uint32(16))
                v = v1_mix * v2_mix
            else:
                v_pre = k_mix ^ (i * jnp.uint32(2654435761))
                v_pre = v_pre ^ (rows * jnp.uint32(1234567891))
                v_pre = v_pre ^ (cols * jnp.uint32(987654321))
                
                v_mix = (v_pre ^ (v_pre >> jnp.uint32(16))) * jnp.uint32(0xBF58476D)
                v_mix = (v_mix ^ (v_mix >> jnp.uint32(13))) * jnp.uint32(0x94D049BB)
                v_mix = v_mix ^ (v_mix >> jnp.uint32(16))
                v = v_mix
                
            x = v.astype(jnp.int8)

            for r in range(self.R):
                W_r = weights_ref[r, :, :]
                acc = pl.dot(x, x, out_dtype=jnp.int32) + W_r.astype(jnp.int32)
                
                acc_u32 = acc.view(jnp.uint32)
                acc_u32 = acc_u32 * jnp.uint32(0x94D049BB)
                mixed = acc_u32 ^ (acc_u32 >> jnp.uint32(16))
                
                x = mixed.astype(jnp.int8)
            pl.store(out_ref, (pl.program_id(0), pl.dslice(T), pl.dslice(T)), x)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, T, T), jnp.int8),
            grid=(N,),
            in_specs=[
                pl.BlockSpec((), lambda i: ()),
                pl.BlockSpec((self.R, T, T), lambda i: (0, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, T, T), lambda i: (i, 0, 0)),
        )(key_mix_arr, weight_matrices.astype(jnp.int32))

    def apply_perturbation(self, weight_matrices, perturbation):
        # Same pattern — one grid program per round
        R, T = self.R, self.T

        def kernel(w_ref, p_ref, out_ref):
            r = pl.program_id(0)
            W = pl.load(w_ref, (r, pl.dslice(T), pl.dslice(T)))
            P = p_ref[...]
            out = pl.dot(W, W) + P
            pl.store(out_ref, (r, pl.dslice(T), pl.dslice(T)), out)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((R, T, T), jnp.uint32),
            grid=(R,),
            in_specs=[
                pl.BlockSpec((1, T, T), lambda r: (r, 0, 0)),
                pl.BlockSpec((T, T), lambda r: (0, 0)),
            ],
            out_specs=pl.BlockSpec((1, T, T), lambda r: (r, 0, 0)),
        )(weight_matrices, perturbation)

    def make_tile(self, key, offset, num_tiles, tile_size, embedding):
        from tyche.algorithm import make_tile
        return make_tile(key, offset, num_tiles, tile_size, embedding)