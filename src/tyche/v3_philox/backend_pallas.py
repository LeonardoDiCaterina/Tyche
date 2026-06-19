import jax.numpy as jnp
import jax
from jax.experimental import pallas as pl

class PallasBackendV3_Philox:
    def __init__(self, num_rounds: int, tile_size: int, word_size: int):
        self.R = num_rounds
        self.T = tile_size
        self.W = word_size

    def hash_parallel(self, tiles_in, weight_matrices):
        N = tiles_in.shape[0]
        T = self.T
        W = self.W
        dtype = jnp.uint32 if W == 32 else jnp.uint64

        tiles_flat = tiles_in.reshape((N, T*T))
        weights_flat = weight_matrices.reshape((self.R, T*T))

        def kernel(tiles_ref, weights_ref, out_ref):
            i = pl.program_id(0)
            total_elements = T * T
            
            if W == 32:
                M0 = jnp.uint32(0xCD9E8D57)
                M1 = jnp.uint32(0xD2511F53)
                chunk_size = total_elements // 4
                
                x0 = pl.load(tiles_ref, (i, pl.dslice(0 * chunk_size, chunk_size)))
                x1 = pl.load(tiles_ref, (i, pl.dslice(1 * chunk_size, chunk_size)))
                x2 = pl.load(tiles_ref, (i, pl.dslice(2 * chunk_size, chunk_size)))
                x3 = pl.load(tiles_ref, (i, pl.dslice(3 * chunk_size, chunk_size)))
                
                for r in range(self.R):
                    k0 = pl.load(weights_ref, (r, pl.dslice(0 * chunk_size, chunk_size)))
                    k1 = pl.load(weights_ref, (r, pl.dslice(1 * chunk_size, chunk_size)))
                    
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
                    
                    x0, x1, x2, x3 = nx0, nx1, nx2, nx3

                pl.store(out_ref, (i, pl.dslice(0 * chunk_size, chunk_size)), x0)
                pl.store(out_ref, (i, pl.dslice(1 * chunk_size, chunk_size)), x1)
                pl.store(out_ref, (i, pl.dslice(2 * chunk_size, chunk_size)), x2)
                pl.store(out_ref, (i, pl.dslice(3 * chunk_size, chunk_size)), x3)
            else:
                M0 = jnp.uint64(0xD2B74407B1CE6E93)
                chunk_size = total_elements // 2
                
                x0 = pl.load(tiles_ref, (i, pl.dslice(0 * chunk_size, chunk_size)))
                x1 = pl.load(tiles_ref, (i, pl.dslice(1 * chunk_size, chunk_size)))
                
                for r in range(self.R):
                    k0 = pl.load(weights_ref, (r, pl.dslice(0 * chunk_size, chunk_size)))
                    
                    lo0 = x0 * M0
                    a_lo = (x0 & 0xFFFFFFFF).astype(jnp.uint64)
                    a_hi = (x0 >> 32).astype(jnp.uint64)
                    b_lo = (M0 & 0xFFFFFFFF).astype(jnp.uint64)
                    b_hi = (M0 >> 32).astype(jnp.uint64)
                    lo_lo = a_lo * b_lo
                    hi_lo = a_hi * b_lo
                    lo_hi = a_lo * b_hi
                    hi_hi = a_hi * b_hi
                    cross = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFF) + (lo_hi & 0xFFFFFFFF)
                    hi0 = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (cross >> 32)
                    
                    nx0 = hi0 ^ k0 ^ x1
                    nx1 = lo0
                    
                    x0, x1 = nx0, nx1
                    
                pl.store(out_ref, (i, pl.dslice(0 * chunk_size, chunk_size)), x0)
                pl.store(out_ref, (i, pl.dslice(1 * chunk_size, chunk_size)), x1)

        out_flat = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, T*T), dtype),
            grid=(N,),
            in_specs=[
                pl.BlockSpec((1, T*T), lambda i: (i, 0)),
                pl.BlockSpec((self.R, T*T), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((1, T*T), lambda i: (i, 0)),
        )(tiles_flat, weights_flat)
        return out_flat.reshape((N, T, T))

    def make_tiles(self, key, offset, num_tiles, tile_size, embedding, word_size):
        from tyche.v3_philox.algorithm import make_tiles
        return make_tiles(key, offset, num_tiles, tile_size, embedding, word_size)
