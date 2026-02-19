"""
Contract tests for the split function.
A valid split implementation must:
- Return exactly num keys of the correct shape
- Produce keys that are all distinct from each other
- Produce keys distinct from the parent
- Be deterministic (same key + same num → same children)
- Be consistent across different split sizes
"""

import jax
import jax.numpy as jnp
import pytest
from tyche import impl


def make_key(seed=42):
    return jax.random.key(seed, impl=impl)


@pytest.mark.parametrize("num", [2, 4, 8, 16, 100])
def test_split_output_shape(num):
    """split(key, num) must return an array of shape (num,)."""
    key = make_key()
    keys = jax.random.split(key, num=num)
    assert keys.shape == (num,)


def test_split_children_are_distinct():
    """All child keys from a single split must be unique."""
    key = make_key()
    keys = jax.random.split(key, num=16)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not jnp.array_equal(keys[i], keys[j]), (
                f"Child keys at index {i} and {j} are identical"
            )


def test_split_children_differ_from_parent():
    """No child key should equal the parent key."""
    key = make_key()
    children = jax.random.split(key, num=8)
    for i, child in enumerate(children):
        assert not jnp.array_equal(child, key), (
            f"Child key at index {i} equals the parent key"
        )


def test_split_is_deterministic():
    """Same key and num must always produce the same children."""
    key = make_key()
    children1 = jax.random.split(key, num=8)
    children2 = jax.random.split(key, num=8)
    assert jnp.array_equal(children1, children2)


def test_split_different_parents_produce_different_children():
    """Keys from different parents should not collide."""
    key1 = make_key(seed=1)
    key2 = make_key(seed=2)
    c1 = jax.random.split(key1, num=8)
    c2 = jax.random.split(key2, num=8)
    # Check no child from one parent equals any child from the other
    for i in range(8):
        for j in range(8):
            assert not jnp.array_equal(c1[i], c2[j]), (
                "Children from different parents share keys"
            )


def test_split_2_matches_default_split():
    """jax.random.split(key) with no num arg defaults to 2 children."""
    key = make_key()
    children = jax.random.split(key)
    assert children.shape == (2,)