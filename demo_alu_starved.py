import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import argparse
import time
import functools
import numpy as np

FAST_MUL1 = 0xBF58476D
FAST_MUL2 = 0x94D049BB
MAX_VAL = 4294967295.0

def fast_mix_u32(x):
    x = (x ^ (x >> jnp.uint32(16))) * jnp.uint32(FAST_MUL1)
    x = (x ^ (x >> jnp.uint32(13))) * jnp.uint32(FAST_MUL2)
    x = x ^ (x >> jnp.uint32(16))
    return x

# ==============================================================================
# ALU-STARVED PHYSICS KERNEL
# ==============================================================================
def pallas_physics_kernel(key_mix_ref, weights_ref, out_ref):
    i = pl.program_id(0)
    tile_L = jnp.uint32(i * 2)
    tile_R = jnp.uint32(i * 2 + 1)
    key_mix = key_mix_ref[...]
    rows = jnp.arange(16, dtype=jnp.uint32)[:, None]
    cols = jnp.arange(16, dtype=jnp.uint32)[None, :]
    
    # ---------------------------------------------------------
    # 1. Tyche Tensor Core Generation
    # ---------------------------------------------------------
    vL = fast_mix_u32(key_mix ^ (tile_L * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
    vR = fast_mix_u32(key_mix ^ (tile_R * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
    
    R_i8 = vR.astype(jnp.int8)
    L_out = vL + pl.dot(R_i8, R_i8).astype(jnp.uint32)
    L_i8 = L_out.astype(jnp.int8)
    R_out = vR + pl.dot(L_i8, L_i8).astype(jnp.uint32)
    
    x = fast_mix_u32(L_out).astype(jnp.float32) / jnp.float32(MAX_VAL)
    y = fast_mix_u32(R_out).astype(jnp.float32) / jnp.float32(MAX_VAL)
    
    # ---------------------------------------------------------
    # 2. Heavy ALU Physics Simulation Loop
    # ---------------------------------------------------------
    # We run 20 iterations of transcendental functions to completely 
    # saturate the GPU's standard math ALUs.
    for _ in range(20):
        # sin, cos, exp all map to expensive math-library PTX instructions
        x_new = jnp.sin(x * 10.0) + jnp.cos(y * 10.0)
        y = jnp.exp(-jnp.abs(x * y))
        x = x_new
    
    out_ref[0] = jnp.sum(x)

@functools.partial(jax.jit, static_argnames=['num_points'])
def jax_native_physics(key, num_points):
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, shape=(num_points,))
    y = jax.random.uniform(k2, shape=(num_points,))
    
    # 20 iterations of transcendental functions
    for _ in range(20):
        x_new = jnp.sin(x * 10.0) + jnp.cos(y * 10.0)
        y = jnp.exp(-jnp.abs(x * y))
        x = x_new
        
    return jnp.sum(x)


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================
def run_benchmark(total_points_per_run, num_runs):
    print("\n" + "="*80)
    print(f"THE ALU-STARVED BENCHMARK: TYCHE PALLAS vs NATIVE THREEFRY")
    print(f"Points per run: {total_points_per_run:,} | Total runs: {num_runs}")
    print("="*80)
    
    key_mix = jnp.array([123456789], dtype=jnp.uint32)
    weights = jnp.zeros((1, 16), dtype=jnp.uint32)
    
    # Our block processes 256 points (x and y)
    num_blocks = total_points_per_run // 256
    pallas_physics = jax.jit(pl.pallas_call(
        pallas_physics_kernel, 
        out_shape=jax.ShapeDtypeStruct((num_blocks,), jnp.float32), 
        grid=(num_blocks,), 
        in_specs=[pl.BlockSpec((1,), lambda i: (0,)), pl.BlockSpec((1, 16), lambda i: (0, 0))], 
        out_specs=pl.BlockSpec((1,), lambda i: (i,))
    ))
    
    # Warmup
    print("Warming up JIT compilers...")
    pallas_physics(key_mix, weights).block_until_ready()
    jax_native_physics(jax.random.PRNGKey(42), total_points_per_run).block_until_ready()
    
    # Run Tyche Pallas
    print("Benchmarking Tyche Pallas (Tensor Core)...")
    t0 = time.perf_counter()
    for _ in range(num_runs): 
        pallas_physics(key_mix, weights).block_until_ready()
    t_pallas = (time.perf_counter() - t0) / num_runs
    
    # Run Native Threefry
    print("Benchmarking JAX Native Threefry (ALU)...")
    t0 = time.perf_counter()
    for _ in range(num_runs): 
        jax_native_physics(jax.random.PRNGKey(42), total_points_per_run).block_until_ready()
    t_native = (time.perf_counter() - t0) / num_runs
    
    total_val = jnp.sum(pallas_physics(key_mix, weights))
    est = total_val / float(num_blocks * 256)
    
    print("\n" + "-"*80)
    print(f"  Estimate (Physics Score): {est:.6f}")
    print(f"  Native Threefry: {t_native:.5f}s per run")
    print(f"  Tyche Pallas:    {t_pallas:.5f}s per run")
    print(f"  Speedup:         {t_native / t_pallas:.2f}x")
    print("-"*80)
    
    if (t_native / t_pallas) > 1.0:
        print("\nSUCCESS: Hardware Concurrency achieved! Tyche is FASTER because Tensor Cores handled the PRNG while ALUs were saturated by the physics loop.")
    else:
        print("\nRESULT: Native Threefry still won, meaning ALU throughput is so high that Tensor Core packing overhead is still the limiting factor.")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 50 million points * 50 runs to ensure we heavily stress the ALUs
    parser.add_argument("--total", type=int, default=50_000_000)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    
    run_benchmark(args.total, args.runs)
