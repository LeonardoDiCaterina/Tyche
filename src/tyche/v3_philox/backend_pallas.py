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

        def kernel(tiles_ref, weights_ref, out_ref):
            i = pl.program_id(0)
            x = pl.load(tiles_ref, (i, pl.dslice(T), pl.dslice(T)))
            
            # Flatten to 1D for Philox operations
            total_elements = T * T
            x_flat = x.reshape(-1)
            
            if W == 32:
                M0 = jnp.uint32(0xCD9E8D57)
                M1 = jnp.uint32(0xD2511F53)
                chunk_size = total_elements // 4
                
                for r in range(self.R):
                    W_r = pl.load(weights_ref, (r, pl.dslice(T), pl.dslice(T))).reshape(-1)
                    k0 = W_r[0:chunk_size]
                    k1 = W_r[chunk_size:2*chunk_size]
                    
                    x0 = x_flat[0:chunk_size]
                    x1 = x_flat[chunk_size:2*chunk_size]
                    x2 = x_flat[2*chunk_size:3*chunk_size]
                    x3 = x_flat[3*chunk_size:4*chunk_size]
                    
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
                    
                    x_flat = jnp.concatenate([nx0, nx1, nx2, nx3])
            else:
                M0 = jnp.uint64(0xD2B74407B1CE6E93)
                chunk_size = total_elements // 2
                
                for r in range(self.R):
                    W_r = pl.load(weights_ref, (r, pl.dslice(T), pl.dslice(T))).reshape(-1)
                    k0 = W_r[0:chunk_size]
                    
                    x0 = x_flat[0:chunk_size]
                    x1 = x_flat[chunk_size:2*chunk_size]
                    
                    lo0 = x0 * M0
                    # emulate mulhi64
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
                    
                    x_flat = jnp.concatenate([nx0, nx1])
                    
            x_out = x_flat.reshape((T, T))
            pl.store(out_ref, (i, pl.dslice(T), pl.dslice(T)), x_out)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, T, T), dtype),
            grid=(N,),
            in_specs=[
                pl.BlockSpec((1, T, T), lambda i: (i, 0, 0)),
                pl.BlockSpec((self.R, T, T), lambda i: (0, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, T, T), lambda i: (i, 0, 0)),
        )(tiles_in, weight_matrices)

    def make_tiles(self, key, offset, num_tiles, tile_size, embedding):
        from tyche.v3_philox.algorithm import make_tiles
        return make_tiles(key, offset, num_tiles, tile_size, embedding, self.W)
