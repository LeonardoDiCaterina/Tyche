import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np

# Import Tyche PRNGs
from tyche.v5b_bijective.config import TycheV5bConfig

# ---------------------------------------------------------
# 1. Pi Monte Carlo Estimation Kernel
# ---------------------------------------------------------
# We generate N points (x, y) in [0, 1] and count how many fall inside the unit circle.
def monte_carlo_pi_step(key, points_per_step):
    # We need 2 * points_per_step random floats
    key_x, key_y = jax.random.split(key)
    x = jax.random.uniform(key_x, shape=(points_per_step,))
    y = jax.random.uniform(key_y, shape=(points_per_step,))
    
    inside_circle = (x**2 + y**2) <= 1.0
    return jnp.sum(inside_circle)

monte_carlo_pi_step = jax.jit(monte_carlo_pi_step, static_argnames=['points_per_step'])

# ---------------------------------------------------------
# 2. Benchmark Logic
# ---------------------------------------------------------
def run_benchmark(total_points, chunk_size):
    num_chunks = total_points // chunk_size
    print("="*70)
    print(f"Monte Carlo Pi Estimation (Real-Life Workload)")
    print(f"Total Points: {total_points:,} | Chunks: {num_chunks} of {chunk_size:,}")
    print("="*70)
    
    # 1. Benchmark JAX Default (Threefry)
    print("\n[1] Benchmarking JAX Default (Threefry)...")
    try:
        key_threefry = jax.random.PRNGKey(42)
        
        # Warmup
        monte_carlo_pi_step(key_threefry, chunk_size).block_until_ready()
        
        t0 = time.perf_counter()
        total_inside = 0
        current_key = key_threefry
        for _ in range(num_chunks):
            current_key, subkey = jax.random.split(current_key)
            total_inside += monte_carlo_pi_step(subkey, chunk_size).block_until_ready()
        t1 = time.perf_counter()
        
        pi_estimate = 4.0 * (total_inside / total_points)
        time_threefry = t1 - t0
        print(f"    Pi Estimate: {pi_estimate:.6f}")
        print(f"    Time taken:  {time_threefry:.4f} seconds")
    except Exception as e:
        print(f"    Failed: {e}")
        time_threefry = float('inf')

    # 2. Benchmark Tyche V5.b (Tensor Core Bijective)
    print("\n[2] Benchmarking Tyche V5.b (Tensor Core Bijective)...")
    try:
        cfg_v5b = TycheV5bConfig(tile_size=16, num_rounds=1, backend="cuda")
        impl_v5b = cfg_v5b.build()
        key_v5b = jax.random.key(42, impl=impl_v5b)
        
        # Warmup
        monte_carlo_pi_step(key_v5b, chunk_size).block_until_ready()
        
        t0 = time.perf_counter()
        total_inside = 0
        current_key = key_v5b
        for _ in range(num_chunks):
            current_key, subkey = jax.random.split(current_key)
            total_inside += monte_carlo_pi_step(subkey, chunk_size).block_until_ready()
        t1 = time.perf_counter()
        
        pi_estimate = 4.0 * (total_inside / total_points)
        time_v5b = t1 - t0
        print(f"    Pi Estimate: {pi_estimate:.6f}")
        print(f"    Time taken:  {time_v5b:.4f} seconds")
        
        print("\n" + "="*70)
        speedup = time_threefry / time_v5b
        print(f"RESULT: Tyche V5.b is {speedup:.2f}x faster than JAX Native Threefry!")
        print("="*70)
    except Exception as e:
        print(f"    Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 5 Billion points by default!
    parser.add_argument("--total", type=int, default=5_000_000_000)
    parser.add_argument("--chunk", type=int, default=100_000_000)
    args = parser.parse_args()
    
    run_benchmark(args.total, args.chunk)
