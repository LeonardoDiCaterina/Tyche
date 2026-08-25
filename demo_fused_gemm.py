import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import argparse
import time
import functools
import numpy as np

FAST_MUL1 = 0xBF58476D
FAST_MUL2 = 0x94D049BB

def fast_mix_u32(x):
    x = (x ^ (x >> jnp.uint32(16))) * jnp.uint32(FAST_MUL1)
    x = (x ^ (x >> jnp.uint32(13))) * jnp.uint32(FAST_MUL2)
    x = x ^ (x >> jnp.uint32(16))
    return x

# ==============================================================================
# FUSED RANDOMIZED GEMM KERNEL (TYCHE PALLAS)
# ==============================================================================
# X is (16, K) block. We compute a (16, 16) output block of Y.
# We iterate over K // 16 blocks to accumulate the dot product.
def fused_gemm_kernel(X_ref, key_mix_ref, Y_ref):
    row = pl.program_id(0)
    col = pl.program_id(1)
    
    key_mix = key_mix_ref[...]
    acc = jnp.zeros((16, 16), dtype=jnp.int32)
    
    rows = jnp.arange(16, dtype=jnp.uint32)[:, None]
    cols = jnp.arange(16, dtype=jnp.uint32)[None, :]
    
    # We statically assume K = 1024 (64 blocks of 16)
    for k in range(64):
        # 1. Load X block (16x16) from global memory
        x_block = pl.load(X_ref, (pl.dslice(0, 16), pl.dslice(k * 16, 16))).astype(jnp.int8)
        
        # 2. Tyche Tensor Core PRNG (Generate random W block directly in registers)
        tile_L = jnp.uint32(row * 64 + k)
        tile_R = jnp.uint32(k * 64 + col)
        
        vL = fast_mix_u32(key_mix ^ (tile_L * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
        vR = fast_mix_u32(key_mix ^ (tile_R * jnp.uint32(2654435761)) ^ (rows * jnp.uint32(1234567891)) ^ (cols * jnp.uint32(987654321)))
        
        R_i8 = vR.astype(jnp.int8)
        L_out = vL + pl.dot(R_i8, R_i8).astype(jnp.uint32)
        W_block = L_out.astype(jnp.int8) # The random weight matrix block
        
        # 3. Multiply X block with random W block using Tensor Cores
        acc += pl.dot(x_block, W_block).astype(jnp.int32)
        
    Y_ref[...] = acc


# ==============================================================================
# JAX NATIVE THREEFRY (UNFUSED GEMM)
# ==============================================================================
@functools.partial(jax.jit, static_argnames=['M', 'K', 'N'])
def jax_native_gemm(key, X, M, K, N):
    # JAX cannot fuse randint with cuBLAS gemm.
    # W is generated and written to global memory.
    W = jax.random.randint(key, shape=(K, N), minval=-128, maxval=127, dtype=jnp.int8)
    
    # cuBLAS reads X and W from global memory
    # We use dot_general to ensure it maps to integer tensor cores if available
    Y = jax.lax.dot_general(X, W, (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
    return Y


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================
def run_benchmark(num_runs):
    M, K, N = 1024, 1024, 1024
    print("\n" + "="*80)
    print(f"THE HOLY GRAIL: FUSED RANDOMIZED GEMM")
    print(f"Matrix Sizes: X[{M},{K}] @ W[{K},{N}] | Total runs: {num_runs}")
    print("="*80)
    
    # Input data matrix
    key_x = jax.random.PRNGKey(0)
    X = jax.random.randint(key_x, shape=(M, K), minval=-128, maxval=127, dtype=jnp.int8)
    
    key_mix = jnp.array([123456789], dtype=jnp.uint32)
    
    grid = (M // 16, N // 16)
    pallas_gemm = jax.jit(pl.pallas_call(
        fused_gemm_kernel, 
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.int32), 
        grid=grid, 
        in_specs=[
            pl.BlockSpec((16, K), lambda i, j: (i, 0)), # X block (row mapped)
            pl.BlockSpec((1,), lambda i, j: (0,))       # key_mix
        ], 
        out_specs=pl.BlockSpec((16, 16), lambda i, j: (i, j)) # Y block
    ))
    
    # Warmup
    print("Warming up JIT compilers...")
    pallas_gemm(X, key_mix).block_until_ready()
    jax_native_gemm(jax.random.PRNGKey(42), X, M, K, N).block_until_ready()
    
    # Run Tyche Pallas
    print("Benchmarking Tyche Pallas (Fused Tensor Core PRNG + GEMM)...")
    t0 = time.perf_counter()
    for _ in range(num_runs): 
        pallas_gemm(X, key_mix).block_until_ready()
    t_pallas = (time.perf_counter() - t0) / num_runs
    
    # Run Native Threefry
    print("Benchmarking JAX Native Threefry (Unfused GEMM)...")
    t0 = time.perf_counter()
    for _ in range(num_runs): 
        jax_native_gemm(jax.random.PRNGKey(42), X, M, K, N).block_until_ready()
    t_native = (time.perf_counter() - t0) / num_runs
    
    print("\n" + "-"*80)
    print(f"  Native Threefry (Unfused): {t_native:.5f}s per run")
    print(f"  Tyche Pallas (Fused):      {t_pallas:.5f}s per run")
    print(f"  Speedup:                   {t_native / t_pallas:.2f}x")
    print("-"*80)
    
    if (t_native / t_pallas) > 1.0:
        print("\nVICTORY! Fusing the PRNG directly into the GEMM kernel bypassed global memory, allowing Tyche to crush Native Threefry!")
    else:
        print("\nRESULT: Native Threefry still won. JAX's cuBLAS integration is too highly optimized.")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    
    run_benchmark(args.runs)
