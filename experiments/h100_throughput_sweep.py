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

# Import Tyche V2.1
from tyche.v2_1.config import TycheV2_1Config

def run_sweep(batch_sizes, num_warmups=3, num_iters=10):
    results = []
    
    threefry_impl = jax_prng.threefry_prng_impl
    threefry_key = jax.random.key(42, impl=threefry_impl)
    tyche_v1_key = jax.random.key(42, impl=tyche_v1_impl)
    
    for batch_size in batch_sizes:
        print(f"\n--- Running Sweep for Batch Size: {batch_size} ---")
        
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
        print(f"Threefry (T=N/A, R=N/A): {throughput_gbps:.2f} GB/s")

        # 2. Benchmark Tyche V1
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
        print(f"Tyche V1 (T=4, R=4): {throughput_gbps:.2f} GB/s")

        # 3. Benchmark Tyche V2 configurations
        for backend_name in ["pallas", "cuda"]:
            for tile_size in [16, 32, 64]:
                if backend_name == "cuda" and tile_size != 16:
                    continue # WMMA MVP only supports T=16
                for num_rounds in [2, 4, 6, 8]:
                    cfg = TycheV2Config(tile_size=tile_size, num_rounds=num_rounds, backend=backend_name)
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
                        "generator": f"Tyche V2 ({backend_name})", "tile_size": tile_size, "num_rounds": num_rounds, "batch_size": batch_size, "throughput_GBs": throughput_gbps
                    })
                    print(f"Tyche V2 {backend_name} (T={tile_size}, R={num_rounds}): {throughput_gbps:.2f} GB/s")
                except Exception as e:
                    print(f"Failed configuration {backend_name} T={tile_size}, R={num_rounds}: {e}")

        # 4. Benchmark Tyche V2.1 configurations (vectorized)
        for tile_size in [16, 32, 64]:
            for num_rounds in [2, 4, 6, 8]:
                cfg = TycheV2_1Config(tile_size=tile_size, num_rounds=num_rounds, backend="pallas")
                impl = cfg.build()
                key = jax.random.key(42, impl=impl)
                
                def gen_tyche_v2_1():
                    return jax.random.bits(key, shape=(batch_size,), dtype=jnp.uint32)
                f_tyche_v2_1 = jax.jit(gen_tyche_v2_1)
                
                # Warmup
                try:
                    for _ in range(num_warmups): f_tyche_v2_1().block_until_ready()
                    
                    t0 = time.perf_counter()
                    for _ in range(num_iters): f_tyche_v2_1().block_until_ready()
                    t1 = time.perf_counter()
                    
                    throughput_gbps = (batch_size * 4 * num_iters) / (t1 - t0) / 1e9
                    results.append({
                        "generator": "Tyche V2.1", "tile_size": tile_size, "num_rounds": num_rounds, "batch_size": batch_size, "throughput_GBs": throughput_gbps
                    })
                    print(f"Tyche V2.1 (T={tile_size}, R={num_rounds}): {throughput_gbps:.2f} GB/s")
                except Exception as e:
                    print(f"Failed configuration T={tile_size}, R={num_rounds}: {e}")

        # 5. Benchmark Tyche V3 Philox
        from tyche.v3_philox.config import TycheV3_PhiloxConfig
        for word_size in [32, 64]:
            for num_rounds in [2, 4]:
                cfg = TycheV3_PhiloxConfig(tile_size=4, num_rounds=num_rounds, word_size=word_size, backend="pallas")
                impl = cfg.build()
                key = jax.random.key(42, impl=impl)
                
                def gen_tyche_v3():
                    return jax.random.bits(key, shape=(batch_size,), dtype=jnp.uint32)
                f_tyche_v3 = jax.jit(gen_tyche_v3)
                
                try:
                    for _ in range(num_warmups): f_tyche_v3().block_until_ready()
                    t0 = time.perf_counter()
                    for _ in range(num_iters): f_tyche_v3().block_until_ready()
                    t1 = time.perf_counter()
                    
                    throughput_gbps = (batch_size * 4 * num_iters) / (t1 - t0) / 1e9
                    name = f"Philox-{word_size}"
                    results.append({
                        "generator": name, "tile_size": 4, "num_rounds": num_rounds, "batch_size": batch_size, "throughput_GBs": throughput_gbps
                    })
                    print(f"{name} (T=4, R={num_rounds}): {throughput_gbps:.2f} GB/s")
                except Exception as e:
                    print(f"Failed configuration Philox-{word_size} T=4, R={num_rounds}: {e}")

        # 6. Benchmark Tyche V4 Threefry
        from tyche.v4_threefry.config import TycheV4_ThreefryConfig
        for word_size in [32, 64]:
            for num_rounds in [2, 4]:
                cfg = TycheV4_ThreefryConfig(tile_size=4, num_rounds=num_rounds, word_size=word_size, backend="pallas")
                impl = cfg.build()
                key = jax.random.key(42, impl=impl)
                
                def gen_tyche_v4():
                    return jax.random.bits(key, shape=(batch_size,), dtype=jnp.uint32)
                f_tyche_v4 = jax.jit(gen_tyche_v4)
                
                try:
                    for _ in range(num_warmups): f_tyche_v4().block_until_ready()
                    t0 = time.perf_counter()
                    for _ in range(num_iters): f_tyche_v4().block_until_ready()
                    t1 = time.perf_counter()
                    
                    throughput_gbps = (batch_size * 4 * num_iters) / (t1 - t0) / 1e9
                    name = f"Threefry-{word_size}"
                    results.append({
                        "generator": name, "tile_size": 4, "num_rounds": num_rounds, "batch_size": batch_size, "throughput_GBs": throughput_gbps
                    })
                    print(f"{name} (T=4, R={num_rounds}): {throughput_gbps:.2f} GB/s")
                except Exception as e:
                    print(f"Failed configuration Threefry-{word_size} T=4, R={num_rounds}: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=str, default="10000000,50000000,100000000,200000000,500000000", help="Comma-separated batch sizes to benchmark")
    parser.add_argument("--output", type=str, default="results/throughput_results.csv", help="Output CSV path")
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Running H100 Throughput Sweep for Batch Sizes {batch_sizes}...")
    
    results = run_sweep(batch_sizes)
    
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generator", "tile_size", "num_rounds", "batch_size", "throughput_GBs"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Sweep complete. Results saved to {args.output}")
