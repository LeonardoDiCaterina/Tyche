import argparse
import json
import os
import jax
import jax.numpy as jnp
from tyche.v2.config import TycheV2Config

# We reuse the statistical logic implicitly by simulating an avalanche test or simply outputting data to be evaluated.
# For simplicity in this script, we'll implement a basic Strict Avalanche Criterion (SAC) check on the first round outputs.

def measure_avalanche(tile_size, num_rounds, embedding, num_samples=1000):
    cfg = TycheV2Config(tile_size=tile_size, num_rounds=num_rounds, embedding=embedding, backend="jax")
    impl = cfg.build()
    
    # Base key
    base_key = jax.random.key(42, impl=impl)
    
    # Generate base bits
    base_bits = jax.random.bits(base_key, shape=(num_samples,), dtype=jnp.uint32)
    
    # Perturb the key slightly (fold_in a different counter)
    alt_key = jax.random.fold_in(base_key, 1)
    alt_bits = jax.random.bits(alt_key, shape=(num_samples,), dtype=jnp.uint32)
    
    diff = base_bits ^ alt_bits
    
    # Count set bits (bit flip ratio)
    # Convert uint32 array to actual bit representations and count
    bit_flips = sum(bin(int(x)).count('1') for x in diff)
    total_bits = num_samples * 32
    flip_ratio = bit_flips / total_bits
    
    return flip_ratio

def run_convergence_sweep():
    results = {}
    embeddings = ["hash", "diagonal", "row", "rank1"]
    rounds = [1, 2, 3, 4, 5, 6, 8]
    tile_size = 16
    
    for emb in embeddings:
        results[emb] = {}
        for r in rounds:
            ratio = measure_avalanche(tile_size, r, emb)
            results[emb][r] = ratio
            print(f"Embedding: {emb}, Rounds: {r} -> Bit Flip Ratio: {ratio:.4f} (Ideal: 0.5)")
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/convergence_results.json", help="Output JSON path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Running Embedding Convergence Sweep...")
    
    results = run_convergence_sweep()
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Convergence sweep complete. Results saved to {args.output}")
