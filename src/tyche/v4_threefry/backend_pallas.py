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

        tiles_flat = tiles_in.reshape((N, T*T))
        weights_flat = weight_matrices.reshape((self.R, T*T))

        def kernel(tiles_ref, weights_ref, out_ref):
            i = pl.program_id(0)
            total_elements = T * T
            
            if W == 32:
                R_CONSTS = (
                    (10, 26), (11, 21), (13, 27), (23, 5),
                    (6, 20), (17, 11), (25, 10), (18, 20)
                )
                chunk_size = total_elements // 4
                
                x0 = pl.load(tiles_ref, (i, pl.dslice(0 * chunk_size, chunk_size)))
                x1 = pl.load(tiles_ref, (i, pl.dslice(1 * chunk_size, chunk_size)))
                x2 = pl.load(tiles_ref, (i, pl.dslice(2 * chunk_size, chunk_size)))
                x3 = pl.load(tiles_ref, (i, pl.dslice(3 * chunk_size, chunk_size)))
                
                for r in range(self.R):
                    k0 = pl.load(weights_ref, (r, pl.dslice(0 * chunk_size, chunk_size)))
                    k1 = pl.load(weights_ref, (r, pl.dslice(1 * chunk_size, chunk_size)))
                    k2 = pl.load(weights_ref, (r, pl.dslice(2 * chunk_size, chunk_size)))
                    k3 = pl.load(weights_ref, (r, pl.dslice(3 * chunk_size, chunk_size)))
                    
                    x0 = x0 + k0
                    x1 = x1 + k1
                    x2 = x2 + k2
                    x3 = x3 + k3
                    
                    rot1 = R_CONSTS[r % 8][0]
                    rot2 = R_CONSTS[r % 8][1]
                    
                    x0 = x0 + x1
                    x1 = (x1 << rot1) | (x1 >> (32 - rot1))
                    x1 = x1 ^ x0
                    
                    x2 = x2 + x3
                    x3 = (x3 << rot2) | (x3 >> (32 - rot2))
                    x3 = x3 ^ x2
                    
                    x0, x1, x2, x3 = x0, x3, x1, x2

                pl.store(out_ref, (i, pl.dslice(0 * chunk_size, chunk_size)), x0)
                pl.store(out_ref, (i, pl.dslice(1 * chunk_size, chunk_size)), x1)
                pl.store(out_ref, (i, pl.dslice(2 * chunk_size, chunk_size)), x2)
                pl.store(out_ref, (i, pl.dslice(3 * chunk_size, chunk_size)), x3)
            else:
                R_CONSTS = (16, 42, 12, 31, 16, 32, 24, 21)
                chunk_size = total_elements // 2
                
                x0 = pl.load(tiles_ref, (i, pl.dslice(0 * chunk_size, chunk_size)))
                x1 = pl.load(tiles_ref, (i, pl.dslice(1 * chunk_size, chunk_size)))
                
                for r in range(self.R):
                    k0 = pl.load(weights_ref, (r, pl.dslice(0 * chunk_size, chunk_size)))
                    k1 = pl.load(weights_ref, (r, pl.dslice(1 * chunk_size, chunk_size)))
                    
                    x0 = x0 + k0
                    x1 = x1 + k1
                    
                    rot = R_CONSTS[r % 8]
                    
                    x0 = x0 + x1
                    x1 = (x1 << rot) | (x1 >> (64 - rot))
                    x1 = x1 ^ x0
                    
                    x0, x1 = x0, x1
                    
                pl.store(out_ref, (i, pl.dslice(0 * chunk_size, chunk_size)), x0)
                pl.store(out_ref, (i, pl.dslice(1 * chunk_size, chunk_size)), x1)

        out_flat = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, T*T), dtype),
            grid=(N,),
            in_specs=[
                pl.BlockSpec((T*T,), lambda i: (i,)),
                pl.BlockSpec((T*T,), lambda i: (0,)),
            ],
            out_specs=pl.BlockSpec((T*T,), lambda i: (i,)),
        )(tiles_flat, weights_flat)
        return out_flat.reshape((N, T, T))

    def make_tiles(self, key, offset, num_tiles, tile_size, embedding, word_size):
        from tyche.v3_philox.algorithm import make_tiles
        return make_tiles(key, offset, num_tiles, tile_size, embedding, word_size)
