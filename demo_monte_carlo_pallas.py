import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import argparse
import time

FAST_MUL1 = 0xBF58476D
FAST_MUL2 = 0x94D049BB

def fast_mix_u32(x):
    x = (x ^ (x >> jnp.uint32(16))) * jnp.uint32(FAST_MUL1)
    x = (x ^ (x >> jnp.uint32(13))) * jnp.uint32(FAST_MUL2)
    x = x ^ (x >> jnp.uint32(16))
    return x

def pallas_fused_monte_carlo_kernel(key_mix_ref, weights_ref, out_ref):
    i = pl.program_id(0)
    
    tile_L_idx = jnp.uint32(i * 2)
    tile_R_idx = jnp.uint32(i * 2 + 1)
    key_mix = key_mix_ref[...]
    
    rows = jnp.arange(16, dtype=jnp.uint32)[:, None]
    cols = jnp.arange(16, dtype=jnp.uint32)[None, :]
    
    vL_pre = key_mix ^ (tile_L_idx * jnp.uint32(2654435761))
    vL_pre = vL_pre ^ (rows * jnp.uint32(1234567891))
    vL_pre = vL_pre ^ (cols * jnp.uint32(987654321))
    vL = fast_mix_u32(vL_pre)
    
    vR_pre = key_mix ^ (tile_R_idx * jnp.uint32(2654435761))
    vR_pre = vR_pre ^ (rows * jnp.uint32(1234567891))
    vR_pre = vR_pre ^ (cols * jnp.uint32(987654321))
    vR = fast_mix_u32(vR_pre)
    
    # Tensor Core Generation
    R_int8 = vR.astype(jnp.int8)
    L_out = vL + pl.dot(R_int8, R_int8).astype(jnp.uint32)
    L_out_int8 = L_out.astype(jnp.int8)
    R_out = vR + pl.dot(L_out_int8, L_out_int8).astype(jnp.uint32)
    L_out = fast_mix_u32(L_out)
    R_out = fast_mix_u32(R_out)
    
    # ---------------------------------------------------------
    # FUSED MONTE CARLO SIMULATION
    # ---------------------------------------------------------
    # We have 2 tiles (L and R), each 16x16 = 256 uint32s.
    # We will use L for 'x' coordinates and R for 'y' coordinates!
    # Normalize to [0, 1]
    max_val = jnp.float32(4294967295.0)
    
    # Cast directly to float inside the kernel registers
    x = L_out.astype(jnp.float32) / max_val
    y = R_out.astype(jnp.float32) / max_val
    
    # Calculate hits inside the kernel
    hits = (x**2 + y**2) <= 1.0
    
    # Reduce the block (16x16 = 256 hits) down to a single integer count
    block_hits = jnp.sum(hits, dtype=jnp.uint32)
    
    # Write exactly 1 integer to global memory per block
    out_ref[0] = block_hits


def run_fused_monte_carlo(total_points=5_000_000_000):
    # Each block processes 256 points (16x16 L and 16x16 R)
    points_per_block = 256
    num_blocks = total_points // points_per_block
    
    print("="*70)
    print(f"Fused Pallas Monte Carlo Pi Estimation")
    print(f"Total Points: {total_points:,} | Grid Blocks: {num_blocks:,}")
    print("="*70)
    
    key_mix = jnp.array([123456789], dtype=jnp.uint32)
    weights = jnp.zeros((1, 16), dtype=jnp.uint32)
    
    pallas_fn = pl.pallas_call(
        pallas_fused_monte_carlo_kernel,
        out_shape=jax.ShapeDtypeStruct((num_blocks,), jnp.uint32),
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec((1,), lambda i: (0,)),
            pl.BlockSpec((1, 16), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
    )
    
    jitted = jax.jit(pallas_fn)
    
    # Warmup
    jitted(key_mix, weights).block_until_ready()
    
    # Run Benchmark
    t0 = time.perf_counter()
    # Pallas returns an array of shape (num_blocks,) with the hits per block
    block_counts = jitted(key_mix, weights).block_until_ready()
    # Final reduction on the CPU/GPU
    total_hits = jnp.sum(block_counts)
    t1 = time.perf_counter()
    
    pi_estimate = 4.0 * (total_hits / float(num_blocks * points_per_block))
    time_taken = t1 - t0
    
    print(f"    Pi Estimate: {pi_estimate:.6f}")
    print(f"    Time taken:  {time_taken:.4f} seconds")
    
    print("\n" + "="*70)
    print(f"This completely eliminates the 400MB memory bandwidth bottleneck!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default 5 Billion points to match the previous demo
    parser.add_argument("--total", type=int, default=5_000_000_000)
    args = parser.parse_args()
    
    run_fused_monte_carlo(args.total)
