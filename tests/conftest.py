"""
Central fixtures shared across all test modules.
All tests interact with Tyche through these fixtures — swapping the
implementation never requires changing any test file.
"""

from xxlimited import Str
import pytest
import jax
import jax.numpy as jnp
from jax._src import prng as jax_prng
jax.config.update("jax_enable_x64", True)  # required for uint64 arithmetic

from tyche import impl as tyche_impl


# ---------------------------------------------------------------------------
# Implementation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def impl():
    """The Tyche PRNGImpl under test."""
    return tyche_impl


@pytest.fixture(scope="session")
def reference_impl():
    """JAX's Threefry implementation as behavioral ground truth."""
    return jax_prng.threefry_prng_impl


# ---------------------------------------------------------------------------
# Seed fixtures
# ---------------------------------------------------------------------------

STANDARD_SEEDS = [0, 1, 42, 12345, 2**31 - 1, 2**32 - 1]

@pytest.fixture(params=STANDARD_SEEDS, ids=[f"seed={s}" for s in STANDARD_SEEDS])
def seed(request: pytest.FixtureRequest):
    """Parametrized over a range of representative seed values."""
    return request.param


@pytest.fixture
def default_seed():
    """A single stable seed for non-parametrized tests."""
    return 42


# ---------------------------------------------------------------------------
# Key fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_key(impl, default_seed) -> jnp.ndarray:
    """A single Tyche key built from the default seed."""
    return jax.random.key(default_seed, impl=impl)


@pytest.fixture
def reference_key(reference_impl, default_seed) -> jnp.ndarray:
    """A single Threefry key built from the same default seed."""
    return jax.random.key(default_seed, impl=reference_impl)


@pytest.fixture
def split_keys(sample_key) -> jnp.ndarray:
    """A small batch of keys produced by splitting the sample key."""
    return jax.random.split(sample_key, num=8)


# ---------------------------------------------------------------------------
# Sample size fixtures  (used by statistical tests)
# ---------------------------------------------------------------------------

@pytest.fixture(params=[1_000, 10_000], ids=["n=1k", "n=10k"])
def sample_size(request):
    return request.param