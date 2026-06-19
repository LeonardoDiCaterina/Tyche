import time
import argparse
import jax
import jax.numpy as jnp
from jax._src import prng as jax_prng

from tyche.v1.config import TycheV1Config
from tyche.v2.config import TycheV2Config
from tyche.v2_1.config import TycheV2_1Config
from tyche.v3_philox.config import TycheV3_PhiloxConfig
from tyche.v4_threefry.config import TycheV4_ThreefryConfig

def run_ga_step(key, population, mutation_rate=0.01):
    key_cross, key_mut, key_noise = jax.random.split(key, 3)
    
    # 1. Uniform Crossover: Swap genes with a rolled version of the population
    cross_mask = jax.random.bits(key_cross, shape=population.shape, dtype=jnp.uint32) & 1
    paired = jnp.roll(population, shift=1, axis=0)
    offspring = jnp.where(cross_mask, population, paired)
    
    # 2. Mutation: Add small uniform noise to genes based on mutation rate
    mut_rand = jax.random.uniform(key_mut, shape=population.shape)
    mut_mask = mut_rand < mutation_rate
    
    noise = jax.random.uniform(key_noise, shape=population.shape) - 0.5
    mutated = offspring + mut_mask * noise
    return mutated

def benchmark(impl, name, pop_shape, num_steps=50, num_warmups=5):
    key = jax.random.key(42, impl=impl)
    population = jax.random.uniform(key, shape=pop_shape)
    
    # Compile the step
    step_jit = jax.jit(lambda k, pop: run_ga_step(k, pop))
    
    # Warmup
    try:
        for _ in range(num_warmups):
            key, subkey = jax.random.split(key)
            population = step_jit(subkey, population)
        population.block_until_ready()
    except Exception as e:
        print(f"{name}: FAILED TO RUN ({e})")
        return None
    
    # Measure
    t0 = time.perf_counter()
    for _ in range(num_steps):
        key, subkey = jax.random.split(key)
        population = step_jit(subkey, population)
    population.block_until_ready()
    t1 = time.perf_counter()
    
    avg_step_time = (t1 - t0) / num_steps
    print(f"{name:<35}: {avg_step_time * 1000:>8.3f} ms per GA iteration")
    return avg_step_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=10000, help="Number of individuals")
    parser.add_argument("--genome-len", type=int, default=1000, help="Genes per individual")
    parser.add_argument("--steps", type=int, default=50, help="Benchmark steps")
    args = parser.parse_args()
    
    pop_shape = (args.pop_size, args.genome_len)
    total_elements = args.pop_size * args.genome_len
    print(f"\n--- Genetic Algorithm Benchmark Ablation ---")
    print(f"GA Workload Size: {pop_shape[0]} x {pop_shape[1]} ({total_elements / 1e6:.1f}M elements)")
    print("-" * 55)
    
    # Native JAX implementations
    benchmark(jax_prng.threefry_prng_impl, "Native Threefry (JAX)", pop_shape, num_steps=args.steps)
    if hasattr(jax_prng, "rbg_prng_impl"):
        benchmark(jax_prng.rbg_prng_impl, "Native RBG (JAX)", pop_shape, num_steps=args.steps)
    print("-" * 55)
    
    # Tyche Tensor Core Architectures
    cfg_v1 = TycheV1Config(tile_size=4, num_rounds=4, backend="pallas")
    benchmark(cfg_v1.build(), "Tyche V1 (T=4, R=4) [TensorCore]", pop_shape, num_steps=args.steps)
    
    cfg_v2 = TycheV2Config(tile_size=64, num_rounds=2, backend="pallas")
    benchmark(cfg_v2.build(), "Tyche V2 (T=64, R=2) [TensorCore]", pop_shape, num_steps=args.steps)

    cfg_v21 = TycheV2_1Config(tile_size=64, num_rounds=2, backend="pallas")
    benchmark(cfg_v21.build(), "Tyche V2.1 (T=64, R=2) [TensorCore]", pop_shape, num_steps=args.steps)
    print("-" * 55)
    
    # Tyche ALU Architectures (Ablation to Philox/Threefry)
    cfg_philox_32 = TycheV3_PhiloxConfig(tile_size=4, num_rounds=2, word_size=32, backend="pallas")
    benchmark(cfg_philox_32.build(), "Tyche V3 (Philox-32, R=2) [ALU]", pop_shape, num_steps=args.steps)

    cfg_philox_64 = TycheV3_PhiloxConfig(tile_size=4, num_rounds=2, word_size=64, backend="pallas")
    benchmark(cfg_philox_64.build(), "Tyche V3 (Philox-64, R=2) [ALU]", pop_shape, num_steps=args.steps)

    cfg_threefry_32 = TycheV4_ThreefryConfig(tile_size=4, num_rounds=2, word_size=32, backend="pallas")
    benchmark(cfg_threefry_32.build(), "Tyche V4 (Threefry-32, R=2) [ALU]", pop_shape, num_steps=args.steps)

    cfg_threefry_64 = TycheV4_ThreefryConfig(tile_size=4, num_rounds=2, word_size=64, backend="pallas")
    benchmark(cfg_threefry_64.build(), "Tyche V4 (Threefry-64, R=2) [ALU]", pop_shape, num_steps=args.steps)
    print("-" * 55 + "\n")
