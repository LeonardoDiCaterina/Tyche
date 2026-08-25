import time
import argparse
import jax
import jax.numpy as jnp
from functools import partial

# Import Tyche V5.b Bijective
from tyche.v5b_bijective.config import TycheV5bHybridConfig

# ---------------------------------------------------------
# 1. Define the Objective Function (Rastrigin)
# ---------------------------------------------------------
# The Rastrigin function is a non-convex function often used
# to test optimization algorithms.
@jax.jit
def rastrigin(x):
    A = 10.0
    n = x.shape[-1]
    return A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x), axis=-1)

# ---------------------------------------------------------
# 2. Define the Optimization Algorithm (Random Search)
# ---------------------------------------------------------
def optimize(key, num_particles, num_steps, dim):
    # Initialize particles uniformly in [-5.12, 5.12]
    key, subkey = jax.random.split(key)
    particles = jax.random.uniform(subkey, shape=(num_particles, dim), minval=-5.12, maxval=5.12)
    best_positions = particles
    best_scores = rastrigin(particles)
    
    def step(carry, _):
        current_key, current_best, current_scores = carry
        
        # Split key for noise generation
        current_key, subkey = jax.random.split(current_key)
        
        # Generate random noise (exploration)
        noise = jax.random.normal(subkey, shape=(num_particles, dim)) * 0.1
        proposals = current_best + noise
        
        # Evaluate proposals
        proposal_scores = rastrigin(proposals)
        
        # Update best positions if proposal is better (lower score)
        improved = proposal_scores < current_scores
        next_scores = jnp.where(improved, proposal_scores, current_scores)
        next_best = jnp.where(improved[:, None], proposals, current_best)
        
        return (current_key, next_best, next_scores), None

    # Run the optimization loop
    (_, final_best, final_scores), _ = jax.lax.scan(step, (key, best_positions, best_scores), None, length=num_steps)
    
    # Return the global best found across all particles
    return jnp.min(final_scores)

# ---------------------------------------------------------
# 3. Benchmark Logic
# ---------------------------------------------------------
def run_demo(num_particles=1_000_000, num_steps=100, dim=10):
    print("="*60)
    print(f"Running Random Search Optimization on Rastrigin function")
    print(f"Particles: {num_particles:,} | Steps: {num_steps} | Dimensions: {dim}")
    print("="*60)
    
    # JIT compile the optimization loop
    jitted_optimize = jax.jit(partial(optimize, num_particles=num_particles, num_steps=num_steps, dim=dim))
    
    # --- Benchmark Threefry (Default JAX) ---
    print("\n[1] Benchmarking Default JAX (Threefry)...")
    key_threefry = jax.random.PRNGKey(42)
    
    # Warmup
    jitted_optimize(key_threefry).block_until_ready()
    
    # Time
    t0 = time.perf_counter()
    best_score_threefry = jitted_optimize(key_threefry).block_until_ready()
    t1 = time.perf_counter()
    time_threefry = t1 - t0
    
    print(f"    Time taken: {time_threefry:.4f} seconds")
    print(f"    Best score: {best_score_threefry:.4f}")

    # --- Benchmark Tyche V5.b ---
    print("\n[2] Benchmarking Tyche V5.b (Bijective)...")
    try:
        cfg = TycheV5bHybridConfig(tile_size=16, num_rounds=1, backend="cuda")
        impl = cfg.build()
        key_tyche = jax.random.key(42, impl=impl)
        
        # Warmup
        jitted_optimize(key_tyche).block_until_ready()
        
        # Time
        t0 = time.perf_counter()
        best_score_tyche = jitted_optimize(key_tyche).block_until_ready()
        t1 = time.perf_counter()
        time_tyche = t1 - t0
        
        print(f"    Time taken: {time_tyche:.4f} seconds")
        print(f"    Best score: {best_score_tyche:.4f}")
        
        # Speedup comparison
        speedup = time_threefry / time_tyche
        print("\n" + "="*60)
        print(f"RESULT: Tyche V5.b is {speedup:.2f}x faster than Threefry!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running Tyche V5.b: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dim", type=int, default=10)
    args = parser.parse_args()
    
    run_demo(args.particles, args.steps, args.dim)
