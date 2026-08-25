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
# 1. PI ESTIMATION (2D)
# ==============================================================================
def pallas_pi_kernel(key_mix_ref, weights_ref, out_ref):
    i = pl.program_id(0)
    tile_L = jnp.uint32(i * 2)
    tile_R = jnp.uint32(i * 2 + 1)
    key_mix = key_mix_ref[...]
    rows = jnp.arange(16, dtype=jnp.uint32)[:, None]
    cols = jnp.arange(16, dtype=jnp.uint32)[None, :]
    
    vL = fast_mix_u32(key_mix ^ (tile_L * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
    vR = fast_mix_u32(key_mix ^ (tile_R * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
    
    R_i8 = vR.astype(jnp.int8)
    L_out = vL + pl.dot(R_i8, R_i8).astype(jnp.uint32)
    L_i8 = L_out.astype(jnp.int8)
    R_out = vR + pl.dot(L_i8, L_i8).astype(jnp.uint32)
    
    x = fast_mix_u32(L_out).astype(jnp.float32) / jnp.float32(MAX_VAL)
    y = fast_mix_u32(R_out).astype(jnp.float32) / jnp.float32(MAX_VAL)
    
    hits = (x**2 + y**2) <= 1.0
    out_ref[0] = jnp.sum(hits, dtype=jnp.uint32)

@functools.partial(jax.jit, static_argnames=['num_points'])
def jax_native_pi(key, num_points):
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, shape=(num_points,))
    y = jax.random.uniform(k2, shape=(num_points,))
    return jnp.sum((x**2 + y**2) <= 1.0)


# ==============================================================================
# 2. 1D INTEGRATION: f(x) = x^3 - 2x^2 + x
# True integral over [0, 1] is 1/12 ≈ 0.083333
# ==============================================================================
def pallas_integral_kernel(key_mix_ref, weights_ref, out_ref):
    i = pl.program_id(0)
    tile_L = jnp.uint32(i * 2)
    tile_R = jnp.uint32(i * 2 + 1)
    key_mix = key_mix_ref[...]
    rows = jnp.arange(16, dtype=jnp.uint32)[:, None]
    cols = jnp.arange(16, dtype=jnp.uint32)[None, :]
    
    vL = fast_mix_u32(key_mix ^ (tile_L * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
    vR = fast_mix_u32(key_mix ^ (tile_R * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
    
    R_i8 = vR.astype(jnp.int8)
    L_out = vL + pl.dot(R_i8, R_i8).astype(jnp.uint32)
    L_i8 = L_out.astype(jnp.int8)
    R_out = vR + pl.dot(L_i8, L_i8).astype(jnp.uint32)
    
    # Use L_out and R_out as 512 total samples
    x1 = fast_mix_u32(L_out).astype(jnp.float32) / jnp.float32(MAX_VAL)
    x2 = fast_mix_u32(R_out).astype(jnp.float32) / jnp.float32(MAX_VAL)
    
    f1 = (x1**3) - 2.0*(x1**2) + x1
    f2 = (x2**3) - 2.0*(x2**2) + x2
    
    out_ref[0] = jnp.sum(f1) + jnp.sum(f2)

@functools.partial(jax.jit, static_argnames=['num_points'])
def jax_native_integral(key, num_points):
    x = jax.random.uniform(key, shape=(num_points,))
    f = (x**3) - 2.0*(x**2) + x
    return jnp.sum(f)


# ==============================================================================
# 3. 2D ROSENBROCK FUNCTION
# f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
# We evaluate the average value over [0, 1]x[0, 1]
# ==============================================================================
def pallas_rosenbrock_kernel(key_mix_ref, weights_ref, out_ref):
    i = pl.program_id(0)
    tile_L = jnp.uint32(i * 2)
    tile_R = jnp.uint32(i * 2 + 1)
    key_mix = key_mix_ref[...]
    rows = jnp.arange(16, dtype=jnp.uint32)[:, None]
    cols = jnp.arange(16, dtype=jnp.uint32)[None, :]
    
    vL = fast_mix_u32(key_mix ^ (tile_L * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
    vR = fast_mix_u32(key_mix ^ (tile_R * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
    
    R_i8 = vR.astype(jnp.int8)
    L_out = vL + pl.dot(R_i8, R_i8).astype(jnp.uint32)
    L_i8 = L_out.astype(jnp.int8)
    R_out = vR + pl.dot(L_i8, L_i8).astype(jnp.uint32)
    
    x = fast_mix_u32(L_out).astype(jnp.float32) / jnp.float32(MAX_VAL)
    y = fast_mix_u32(R_out).astype(jnp.float32) / jnp.float32(MAX_VAL)
    
    f = (1.0 - x)**2 + 100.0 * (y - x**2)**2
    out_ref[0] = jnp.sum(f)

@functools.partial(jax.jit, static_argnames=['num_points'])
def jax_native_rosenbrock(key, num_points):
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, shape=(num_points,))
    y = jax.random.uniform(k2, shape=(num_points,))
    f = (1.0 - x)**2 + 100.0 * (y - x**2)**2
    return jnp.sum(f)


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================
def run_benchmarks(total_points_per_run, num_runs):
    print("\n" + "="*80)
    print(f"TYCHE V5.b PALLAS vs JAX NATIVE THREEFRY (100 ITERATION SUITE)")
    print(f"Points per run: {total_points_per_run:,} | Total runs: {num_runs}")
    print("="*80)
    
    key_mix = jnp.array([123456789], dtype=jnp.uint32)
    weights = jnp.zeros((1, 16), dtype=jnp.uint32)
    
    # ------------------
    # 1. PI ESTIMATION
    # ------------------
    print("\n[1] 2D Pi Estimation")
    num_blocks_pi = total_points_per_run // 256
    pallas_pi = jax.jit(pl.pallas_call(pallas_pi_kernel, out_shape=jax.ShapeDtypeStruct((num_blocks_pi,), jnp.uint32), grid=(num_blocks_pi,), in_specs=[pl.BlockSpec((1,), lambda i: (0,)), pl.BlockSpec((1, 16), lambda i: (0, 0))], out_specs=pl.BlockSpec((1,), lambda i: (i,))))
    
    # Warmup
    pallas_pi(key_mix, weights).block_until_ready()
    jax_native_pi(jax.random.PRNGKey(42), total_points_per_run).block_until_ready()
    
    t0 = time.perf_counter()
    for _ in range(num_runs): pallas_pi(key_mix, weights).block_until_ready()
    t_pallas_pi = (time.perf_counter() - t0) / num_runs
    
    t0 = time.perf_counter()
    for _ in range(num_runs): jax_native_pi(jax.random.PRNGKey(42), total_points_per_run).block_until_ready()
    t_native_pi = (time.perf_counter() - t0) / num_runs
    
    # Extract one sample estimate
    total_hits = jnp.sum(pallas_pi(key_mix, weights))
    est_pi = 4.0 * (total_hits / float(num_blocks_pi * 256))
    
    print(f"  Estimate: {est_pi:.6f} (True ~3.14159)")
    print(f"  Native Threefry: {t_native_pi:.5f}s per run")
    print(f"  Tyche Pallas:    {t_pallas_pi:.5f}s per run")
    print(f"  Speedup:         {t_native_pi / t_pallas_pi:.2f}x")

    # ------------------
    # 2. 1D INTEGRATION
    # ------------------
    print("\n[2] 1D Function Integration (f(x) = x^3 - 2x^2 + x)")
    # For integral, our block outputs 512 points
    num_blocks_int = total_points_per_run // 512
    pallas_int = jax.jit(pl.pallas_call(pallas_integral_kernel, out_shape=jax.ShapeDtypeStruct((num_blocks_int,), jnp.float32), grid=(num_blocks_int,), in_specs=[pl.BlockSpec((1,), lambda i: (0,)), pl.BlockSpec((1, 16), lambda i: (0, 0))], out_specs=pl.BlockSpec((1,), lambda i: (i,))))
    
    # Warmup
    pallas_int(key_mix, weights).block_until_ready()
    jax_native_integral(jax.random.PRNGKey(42), total_points_per_run).block_until_ready()
    
    t0 = time.perf_counter()
    for _ in range(num_runs): pallas_int(key_mix, weights).block_until_ready()
    t_pallas_int = (time.perf_counter() - t0) / num_runs
    
    t0 = time.perf_counter()
    for _ in range(num_runs): jax_native_integral(jax.random.PRNGKey(42), total_points_per_run).block_until_ready()
    t_native_int = (time.perf_counter() - t0) / num_runs
    
    total_val = jnp.sum(pallas_int(key_mix, weights))
    est_int = total_val / float(num_blocks_int * 512)
    
    print(f"  Estimate: {est_int:.6f} (True ~0.083333)")
    print(f"  Native Threefry: {t_native_int:.5f}s per run")
    print(f"  Tyche Pallas:    {t_pallas_int:.5f}s per run")
    print(f"  Speedup:         {t_native_int / t_pallas_int:.2f}x")

    # ------------------
    # 3. 2D ROSENBROCK FUNCTION
    # ------------------
    print("\n[3] 2D Rosenbrock Function Evaluation")
    # Our block processes 256 points (x and y)
    num_blocks_ros = total_points_per_run // 256
    pallas_ros = jax.jit(pl.pallas_call(pallas_rosenbrock_kernel, out_shape=jax.ShapeDtypeStruct((num_blocks_ros,), jnp.float32), grid=(num_blocks_ros,), in_specs=[pl.BlockSpec((1,), lambda i: (0,)), pl.BlockSpec((1, 16), lambda i: (0, 0))], out_specs=pl.BlockSpec((1,), lambda i: (i,))))
    
    # Warmup
    pallas_ros(key_mix, weights).block_until_ready()
    jax_native_rosenbrock(jax.random.PRNGKey(42), total_points_per_run).block_until_ready()
    
    t0 = time.perf_counter()
    for _ in range(num_runs): pallas_ros(key_mix, weights).block_until_ready()
    t_pallas_ros = (time.perf_counter() - t0) / num_runs
    
    t0 = time.perf_counter()
    for _ in range(num_runs): jax_native_rosenbrock(jax.random.PRNGKey(42), total_points_per_run).block_until_ready()
    t_native_ros = (time.perf_counter() - t0) / num_runs
    
    total_val_ros = jnp.sum(pallas_ros(key_mix, weights))
    est_ros = total_val_ros / float(num_blocks_ros * 256)
    
    print(f"  Estimate: {est_ros:.6f}")
    print(f"  Native Threefry: {t_native_ros:.5f}s per run")
    print(f"  Tyche Pallas:    {t_pallas_ros:.5f}s per run")
    print(f"  Speedup:         {t_native_ros / t_pallas_ros:.2f}x")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 500 million points * 100 runs = massive workload!
    parser.add_argument("--total", type=int, default=100_000_000)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    
    run_benchmarks(args.total, args.runs)
