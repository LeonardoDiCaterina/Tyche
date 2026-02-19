"""
benchmarks/bench_throughput.py

Throughput benchmarks comparing Tyche against JAX's built-in PRNGs.
Measures raw generation speed across different batch sizes and operations.

Run:
    pytest benchmarks/bench_throughput.py --benchmark-only -v
    pytest benchmarks/bench_throughput.py --benchmark-only --benchmark-sort=mean
    pytest benchmarks/bench_throughput.py --benchmark-only --benchmark-histogram
"""

import pytest
import jax
import jax.numpy as jnp
from jax._src import prng as jax_prng
from tyche import impl as tyche_impl

threefry_impl = jax_prng.threefry_prng_impl

# Pre-built keys — exclude key creation time from generation benchmarks
TYCHE_KEY    = jax.random.key(42, impl=tyche_impl)
THREEFRY_KEY = jax.random.key(42, impl=threefry_impl)

# Warm up JIT before benchmarking
_ = jax.random.uniform(TYCHE_KEY, shape=(1,))
_ = jax.random.uniform(THREEFRY_KEY, shape=(1,))
jax.effects_barrier()

BATCH_SIZES = [1_000, 10_000, 100_000, 1_000_000]


# ---------------------------------------------------------------------------
# random_bits — raw throughput
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.benchmark(group="random_bits")
def test_bench_tyche_random_bits(benchmark, n):
    f = jax.jit(lambda k: jax.random.bits(k, shape=(n,), dtype=jnp.uint32))
    f(TYCHE_KEY).block_until_ready()  # warm up
    benchmark(lambda: f(TYCHE_KEY).block_until_ready())


@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.benchmark(group="random_bits")
def test_bench_threefry_random_bits(benchmark, n):
    f = jax.jit(lambda k: jax.random.bits(k, shape=(n,), dtype=jnp.uint32))
    f(THREEFRY_KEY).block_until_ready()
    benchmark(lambda: f(THREEFRY_KEY).block_until_ready())


# ---------------------------------------------------------------------------
# uniform float generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.benchmark(group="uniform")
def test_bench_tyche_uniform(benchmark, n):
    f = jax.jit(lambda k: jax.random.uniform(k, shape=(n,)))
    f(TYCHE_KEY).block_until_ready()
    benchmark(lambda: f(TYCHE_KEY).block_until_ready())


@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.benchmark(group="uniform")
def test_bench_threefry_uniform(benchmark, n):
    f = jax.jit(lambda k: jax.random.uniform(k, shape=(n,)))
    f(THREEFRY_KEY).block_until_ready()
    benchmark(lambda: f(THREEFRY_KEY).block_until_ready())


# ---------------------------------------------------------------------------
# normal float generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.benchmark(group="normal")
def test_bench_tyche_normal(benchmark, n):
    f = jax.jit(lambda k: jax.random.normal(k, shape=(n,)))
    f(TYCHE_KEY).block_until_ready()
    benchmark(lambda: f(TYCHE_KEY).block_until_ready())


@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.benchmark(group="normal")
def test_bench_threefry_normal(benchmark, n):
    f = jax.jit(lambda k: jax.random.normal(k, shape=(n,)))
    f(THREEFRY_KEY).block_until_ready()
    benchmark(lambda: f(THREEFRY_KEY).block_until_ready())


# ---------------------------------------------------------------------------
# key management — split and fold_in
# ---------------------------------------------------------------------------

@pytest.mark.benchmark(group="split")
def test_bench_tyche_split(benchmark):
    f = jax.jit(lambda k: jax.random.split(k, num=16))
    f(TYCHE_KEY).block_until_ready()
    benchmark(lambda: f(TYCHE_KEY).block_until_ready())


@pytest.mark.benchmark(group="split")
def test_bench_threefry_split(benchmark):
    f = jax.jit(lambda k: jax.random.split(k, num=16))
    f(THREEFRY_KEY).block_until_ready()
    benchmark(lambda: f(THREEFRY_KEY).block_until_ready())


@pytest.mark.benchmark(group="fold_in")
def test_bench_tyche_fold_in(benchmark):
    f = jax.jit(lambda k: jax.random.fold_in(k, 42))
    f(TYCHE_KEY)
    benchmark(lambda: f(TYCHE_KEY))


@pytest.mark.benchmark(group="fold_in")
def test_bench_threefry_fold_in(benchmark):
    f = jax.jit(lambda k: jax.random.fold_in(k, 42))
    f(THREEFRY_KEY)
    benchmark(lambda: f(THREEFRY_KEY))


# ---------------------------------------------------------------------------
# vmap scaling — how well does it scale across batch dimensions?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_keys", [8, 64, 256])
@pytest.mark.benchmark(group="vmap")
def test_bench_tyche_vmap(benchmark, n_keys):
    keys = jax.random.split(TYCHE_KEY, num=n_keys)
    f = jax.jit(jax.vmap(lambda k: jax.random.uniform(k, shape=(1000,))))
    f(keys).block_until_ready()
    benchmark(lambda: f(keys).block_until_ready())


@pytest.mark.parametrize("n_keys", [8, 64, 256])
@pytest.mark.benchmark(group="vmap")
def test_bench_threefry_vmap(benchmark, n_keys):
    keys = jax.random.split(THREEFRY_KEY, num=n_keys)
    f = jax.jit(jax.vmap(lambda k: jax.random.uniform(k, shape=(1000,))))
    f(keys).block_until_ready()
    benchmark(lambda: f(keys).block_until_ready())