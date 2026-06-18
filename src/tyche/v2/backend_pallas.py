import jax.numpy as jnp
import jax
from jax.experimental import pallas as pl

class PallasBackendV2:
    def __init__(self, num_rounds: int, tile_size: int):
        self.R = num_rounds
        self.T = tile_size

    def hash_parallel(self, tiles_int8, weight_matrices):
        N = tiles_int8.shape[0]
        T = self.T

        def kernel(tiles_ref, weights_ref, out_ref):
            # Each grid program handles one tile
            i = pl.program_id(0)
            
            # Load the int8 tile into registers
            x = pl.load(tiles_ref, (i, pl.dslice(T), pl.dslice(T)))
            
            for r in range(self.R):
                W_r = pl.load(weights_ref, (r, pl.dslice(T), pl.dslice(T)))
                
                # Tensor Core Path: int8 * int8 + int32 -> int32
                # Hardware requires K to be 2*T for INT8. We pad with zeros.
                z = jnp.zeros((T, T), dtype=jnp.int8)
                x_lhs = jnp.concatenate([x, z], axis=1)  # (T, 2*T)
                x_rhs = jnp.concatenate([x, z], axis=0)  # (2*T, T)
                
                acc = pl.dot(x_lhs, x_rhs) + W_r
                
                # ALU Path
                # Cast acc to uint32 for bitwise ops, then ODD_MULT
                acc_u32 = acc.view(jnp.uint32)
                acc_u32 = acc_u32 * jnp.uint32(0x94D049BB)
                
                # XOR Fold
                folded = acc_u32 ^ (acc_u32 >> 16)
                
                # Truncate to int8 for the next round (free cast in PTX)
                x = folded.astype(jnp.int8)
                
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
        )(tiles_int8, weight_matrices)

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
