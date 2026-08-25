import jax.numpy as jnp
import jax
from jax.experimental import pallas as pl

class PallasBackendV2:
    def __init__(self, num_rounds: int, tile_size: int):
        self.R = num_rounds
        self.T = tile_size

    def hash_parallel(self, key, offset, num_tiles, weight_matrices, embedding):
        from tyche.v2.algorithm import _mix_key_const
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
                W_r = pl.load(weights_ref, (r, pl.dslice(T), pl.dslice(T)))
                
                acc = pl.dot(x, x)
                acc_u32 = acc.astype(jnp.uint32) + W_r
                
                acc_u32 = acc_u32 * jnp.uint32(0x94D049BB)
                folded = acc_u32 ^ (acc_u32 >> 16)
                
                x = folded.astype(jnp.int8)
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
        )(key_mix_arr, weight_matrices)

    def apply_perturbation(self, weight_matrices, perturbation):
        # Same pattern — one grid program per round, but sizes are T
        R, T = self.R, self.T

        def kernel(w_ref, p_ref, out_ref):
            r = pl.program_id(0)
            W = pl.load(w_ref, (r, pl.dslice(T), pl.dslice(T)))
            P = p_ref[...]
            # JAX treats uint32/int32 as same width, we do full precision 32-bit dot
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
        )(weight_matrices.astype(jnp.uint32), perturbation.astype(jnp.uint32))

    def make_tiles(self, key, offset, num_tiles, tile_size, embedding):
        from tyche.v2.algorithm import make_tiles
        return make_tiles(key, offset, num_tiles, tile_size, embedding)
