import jax.numpy as jnp
import jax
from jax.experimental import pallas as pl

class PallasBackendV4_Threefry:
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
            
            total_elements = T * T
            x_flat = x.reshape(-1)
            
            if W == 32:
                R_CONSTS = jnp.array([
                    [10, 26], [11, 21], [13, 27], [23, 5],
                    [6, 20], [17, 11], [25, 10], [18, 20]
                ], dtype=jnp.uint32)
                chunk_size = total_elements // 4
                
                for r in range(self.R):
                    W_r = pl.load(weights_ref, (r, pl.dslice(T), pl.dslice(T))).reshape(-1)
                    x_flat = x_flat + W_r
                    
                    x0 = x_flat[0:chunk_size]
                    x1 = x_flat[chunk_size:2*chunk_size]
                    x2 = x_flat[2*chunk_size:3*chunk_size]
                    x3 = x_flat[3*chunk_size:4*chunk_size]
                    
                    rot1 = R_CONSTS[r % 8, 0]
                    rot2 = R_CONSTS[r % 8, 1]
                    
                    x0 = x0 + x1
                    x1 = (x1 << rot1) | (x1 >> (32 - rot1))
                    x1 = x1 ^ x0
                    
                    x2 = x2 + x3
                    x3 = (x3 << rot2) | (x3 >> (32 - rot2))
                    x3 = x3 ^ x2
                    
                    x_flat = jnp.concatenate([x0, x3, x1, x2])
            else:
                R_CONSTS = jnp.array([16, 42, 12, 31, 16, 32, 24, 21], dtype=jnp.uint64)
                chunk_size = total_elements // 2
                
                for r in range(self.R):
                    W_r = pl.load(weights_ref, (r, pl.dslice(T), pl.dslice(T))).reshape(-1)
                    x_flat = x_flat + W_r
                    
                    x0 = x_flat[0:chunk_size]
                    x1 = x_flat[chunk_size:2*chunk_size]
                    
                    rot = R_CONSTS[r % 8]
                    
                    x0 = x0 + x1
                    x1 = (x1 << rot) | (x1 >> (64 - rot))
                    x1 = x1 ^ x0
                    
                    x_flat = jnp.concatenate([x0, x1])
                    
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

    def make_tiles(self, key, offset, num_tiles, tile_size, embedding, word_size):
        from tyche.v3_philox.algorithm import make_tiles
        return make_tiles(key, offset, num_tiles, tile_size, embedding, word_size)
