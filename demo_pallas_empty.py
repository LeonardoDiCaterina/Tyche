import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import argparse
import time

# ---------------------------------------------------------
# 1. Pallas Empty Kernel
# ---------------------------------------------------------
# This kernel matches the Tyche V5.b signature but does nothing 
# except fill the output block with zeros.
def empty_kernel(key_mix_ref, weights_ref, out_ref):
    # Get the block index
    i = pl.program_id(0)
    
    # Just write zeros to the output
    out_ref[...] = jnp.zeros((16, 16), dtype=jnp.uint32)

def run_empty_pallas(num_tiles=1000, tile_size=16):
    # Dummy inputs
    key_mix = jnp.array([123456789], dtype=jnp.uint32)
    # V5.b uses 1 round, so weights shape is (1, 16)
    weights = jnp.zeros((1, 16), dtype=jnp.uint32)
    
    print(f"Compiling and running Empty Pallas Kernel with {num_tiles} blocks...")
    
    # Define the Pallas call
    pallas_fn = pl.pallas_call(
        empty_kernel,
        out_shape=jax.ShapeDtypeStruct((num_tiles, tile_size, tile_size), jnp.uint32),
        grid=(num_tiles,),
        in_specs=[
            pl.BlockSpec((1,), lambda i: (0,)),                 # key_mix has shape (1,)
            pl.BlockSpec((1, tile_size), lambda i: (0, 0)),     # weights are shared across all blocks
        ],
        out_specs=pl.BlockSpec((tile_size, tile_size), lambda i: (i, 0, 0)), # Each block outputs a tile_size x tile_size tile
    )
    
    # JIT compile
    jitted_pallas = jax.jit(pallas_fn)
    
    # Run
    t0 = time.perf_counter()
    out = jitted_pallas(key_mix, weights).block_until_ready()
    t1 = time.perf_counter()
    
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"Execution time: {t1 - t0:.4f} seconds")
    print(f"Sample block [0]:\n{out[0]}")
    
    assert jnp.all(out == 0), "Output was not all zeros!"
    print("SUCCESS: Empty Pallas skeleton works!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=int, default=100)
    args = parser.parse_args()
    
    run_empty_pallas(args.tiles)
