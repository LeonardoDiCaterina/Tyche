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

    def hash_parallel(self, tiles, weight_matrices):
        N = tiles.shape[0]
        T = self.T

        def kernel(tiles_ref, weights_ref, out_ref):
            i = pl.program_id(0)
            x = pl.load(tiles_ref, (i, pl.dslice(T), pl.dslice(T)))
            for r in range(self.R):
                W_r = weights_ref[r, :, :]
                acc = pl.dot(x, x, out_dtype=jnp.int32) + W_r
                
                acc_u32 = acc.view(jnp.uint32)
                acc_u32 = acc_u32 * jnp.uint32(0x94D049BB)
                mixed = acc_u32 ^ (acc_u32 >> jnp.uint32(16))
                
                x = mixed.astype(jnp.int8)
            pl.store(out_ref, (i, pl.dslice(T), pl.dslice(T)), x)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, T, T), jnp.int8),
            grid=(N,),
            in_specs=[
                pl.BlockSpec((1, T, T), lambda i: (i, 0, 0)),
                pl.BlockSpec((self.R, T, T), lambda i: (0, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, T, T), lambda i: (i, 0, 0)),
        )(tiles, weight_matrices.astype(jnp.int32))

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