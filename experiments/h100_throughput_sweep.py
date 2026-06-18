import time
import argparse
import csv
import os
import jax
import jax.numpy as jnp
from jax._src import prng as jax_prng

# Import Tyche V1
from tyche import impl as tyche_v1_impl
from tyche.config import TycheConfig

# Import Tyche V2
from tyche.v2.config import TycheV2Config

def run_sweep(batch_size, num_warmups=3, num_iters=10):
    results = []
    
    threefry_impl = jax_prng.threefry_prng_impl
    threefry_key = jax.random.key(42, impl=threefry_impl)
    
    # 1. Benchmark Threefry
    def gen_threefry():
        return jax.random.bits(threefry_key, shape=(batch_size,), dtype=jnp.uint32)
    f_threefry = jax.jit(gen_threefry)
    for _ in range(num_warmups): f_threefry().block_until_ready()
    t0 = time.perf_counter()
    for _ in range(num_iters): f_threefry().block_until_ready()
    t1 = time.perf_counter()
    throughput_gbps = (batch_size * 4 * num_iters) / (t1 - t0) / 1e9
    results.append({
        "generator": "Threefry", "tile_size": "N/A", "num_rounds": "N/A", "batch_size": batch_size, "throughput_GBs": throughput_gbps
    })

    # 2. Benchmark Tyche V1
    tyche_v1_key = jax.random.key(42, impl=tyche_v1_impl)
    def gen_tyche_v1():
        return jax.random.bits(tyche_v1_key, shape=(batch_size,), dtype=jnp.uint32)
    f_tyche_v1 = jax.jit(gen_tyche_v1)
    for _ in range(num_warmups): f_tyche_v1().block_until_ready()
    t0 = time.perf_counter()
    for _ in range(num_iters): f_tyche_v1().block_until_ready()
    t1 = time.perf_counter()
    throughput_gbps = (batch_size * 4 * num_iters) / (t1 - t0) / 1e9
    results.append({
        "generator": "Tyche V1", "tile_size": 4, "num_rounds": 4, "batch_size": batch_size, "throughput_GBs": throughput_gbps
    })

    # 3. Benchmark Tyche V2 configurations
    for tile_size in [16, 32, 64]:
        for num_rounds in [2, 4, 6, 8]:
            cfg = TycheV2Config(tile_size=tile_size, num_rounds=num_rounds, backend="jax")
            impl = cfg.build()
            key = jax.random.key(42, impl=impl)
            
            def gen_tyche_v2():
                return jax.random.bits(key, shape=(batch_size,), dtype=jnp.uint32)
            f_tyche_v2 = jax.jit(gen_tyche_v2)
            
            # Warmup
            try:
                for _ in range(num_warmups): f_tyche_v2().block_until_ready()
                
                t0 = time.perf_counter()
                for _ in range(num_iters): f_tyche_v2().block_until_ready()
                t1 = time.perf_counter()
                
                throughput_gbps = (batch_size * 4 * num_iters) / (t1 - t0) / 1e9
                results.append({
                    "generator": "Tyche V2", "tile_size": tile_size, "num_rounds": num_rounds, "batch_size": batch_size, "throughput_GBs": throughput_gbps
                })
            except Exception as e:
                print(f"Failed configuration T={tile_size}, R={num_rounds}: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10000000, help="Number of uint32s to generate per batch")
    parser.add_argument("--output", type=str, default="results/throughput_results.csv", help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Running H100 Throughput Sweep for Batch Size {args.batch_size}...")
    
    results = run_sweep(args.batch_size)
    
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generator", "tile_size", "num_rounds", "batch_size", "throughput_GBs"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Sweep complete. Results saved to {args.output}")
    for row in results:
        print(f"{row['generator']} (T={row['tile_size']}, R={row['num_rounds']}): {row['throughput_GBs']:.2f} GB/s")
