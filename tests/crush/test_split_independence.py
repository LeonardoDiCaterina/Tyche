"""
A PractRand extension to exercise independence between sibling keys.

The existing "stream.py" script just generates output from a single evolving
key, which is what the standard BigCrush run exercises.  This module
produces two (or more) child streams derived from the same parent, then
interleaves or concatenates them before handing the data to PractRand.  The
purpose is to look for cross-stream correlations that would be invisible to a
single-stream test.

Usage (pytest marker "crush" is required):

    pytest tests/crush/test_split_independence.py -m crush

Because each derived stream uses `jax.random.split` internally, the only
difference between this test and the ordinary one is the mixing of the child
outputs; PractRand itself still sees a uint32 stream on stdin.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from crush.practrand_utils import _require_practrand, _parse_practrand_failures

STREAM = Path(__file__).parent / "stream.py"

pytestmark = pytest.mark.crush


def _run_interleaved(
    n_uint32: int,
    tlmax: str = "32GB",
    parent_seeds: tuple[int, int] = (0, 0),
    indices: tuple[int, int] = (0, 0),
) -> str:
    """Generate two streams and feed them to PractRand.

    * ``parent_seeds`` determines the two initial parent keys.
    * ``indices`` controls the child index folded into each parent; the
      default ``(0,0)`` reproduces the original sibling test where both
      children come from the same parent.

    The streams are interleaved one word at a time. ``n_uint32`` is the
    total length, so each branch contributes half of that amount.
    """

    _require_practrand()

    pycmd = r"""
import os, sys, numpy as np
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import io
_old = sys.stderr
sys.stderr = io.StringIO()
import jax, jax.numpy as jnp
from tyche import impl
sys.stderr = _old

# construct parents and children according to caller parameters
p1 = jax.random.key({seed1}, impl=impl)
p2 = jax.random.key({seed2}, impl=impl)
child1 = jax.random.fold_in(p1, {idx1})
child2 = jax.random.fold_in(p2, {idx2})

count = {half}
chunk = 100_000
out = sys.stdout.buffer
keys = [child1, child2]
written = 0
while written < {total}:
    for i in range(2):
        keys[i], sub = jax.random.split(keys[i])
        cnt = min(chunk, {half} - (written // 2))
        buf = np.array(jax.random.bits(sub, shape=(cnt,), dtype=jnp.uint32), dtype=np.uint32)
        out.write(buf.tobytes()); out.flush()
        written += cnt
""".format(
        seed1=parent_seeds[0],
        seed2=parent_seeds[1],
        idx1=indices[0],
        idx2=indices[1],
        half=n_uint32 // 2,
        total=n_uint32,
    )

    stream_proc = subprocess.Popen([sys.executable, "-c", pycmd], stdout=subprocess.PIPE)
    practrand_proc = subprocess.Popen(
        ["RNG_test", "stdin32", "-tlmax", tlmax],
        stdin=stream_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stream_proc.stdout.close()
    stdout, stderr = practrand_proc.communicate()
    stream_proc.wait()
    return stdout.decode() + stderr.decode()


def _run_interleaved_fold(
    n_uint32: int,
    tlmax: str = "32GB",
    parent_seeds: tuple[int, int] = (0, 0),
    indices: tuple[int, int] = (0, 0),
) -> str:
    """Generate two streams and feed them to PractRand using *fold_in* for
    both the initial child derivation **and** the per-block key advancement.

    The semantics of the arguments are the same as :func:`_run_interleaved`.
    During the inner loop each branch's key is updated by folding in a
    monotonically‑increasing counter so that the stream progresses without
    ever calling ``split``.  This exercises independence properties of
    ``jax.random.fold_in`` both at creation time (different parent/indices)
    and during streaming.
    """

    _require_practrand()

    pycmd = r"""
import os, sys, numpy as np
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import io
_old = sys.stderr
sys.stderr = io.StringIO()
import jax, jax.numpy as jnp
from tyche import impl
sys.stderr = _old

# construct parents and children according to caller parameters
p1 = jax.random.key({seed1}, impl=impl)
p2 = jax.random.key({seed2}, impl=impl)
child1 = jax.random.fold_in(p1, {idx1})
child2 = jax.random.fold_in(p2, {idx2})

count = {half}
chunk = 100_000
out = sys.stdout.buffer
keys = [child1, child2]
counters = [0, 0]
written = 0
while written < {total}:
    for i in range(2):
        # advance by folding in the per‑branch counter
        sub = jax.random.fold_in(keys[i], counters[i])
        counters[i] += 1
        cnt = min(chunk, {half} - (written // 2))
        buf = np.array(jax.random.bits(sub, shape=(cnt,), dtype=jnp.uint32), dtype=np.uint32)
        out.write(buf.tobytes()); out.flush()
        written += cnt
""".format(
        seed1=parent_seeds[0],
        seed2=parent_seeds[1],
        idx1=indices[0],
        idx2=indices[1],
        half=n_uint32 // 2,
        total=n_uint32,
    )

    stream_proc = subprocess.Popen([sys.executable, "-c", pycmd], stdout=subprocess.PIPE)
    practrand_proc = subprocess.Popen(
        ["RNG_test", "stdin32", "-tlmax", tlmax],
        stdin=stream_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stream_proc.stdout.close()
    stdout, stderr = practrand_proc.communicate()
    stream_proc.wait()
    return stdout.decode() + stderr.decode()


def test_fold_independence_small():
    # same-parent siblings but using fold-based advancement at full BigCrush size
    out = _run_interleaved_fold(n_uint32=1 << 35, tlmax="32GB",
                                parent_seeds=(0, 0), indices=(0, 1))
    fails = _parse_practrand_failures(out)
    assert not fails, "Failures when interleaving two fold-derived sibling streams:\n" + "\n".join(fails)


def test_fold_independent_parents_same_index():
    # different parents, same index; advancement still via fold_in counter
    out = _run_interleaved_fold(n_uint32=1 << 35, tlmax="32GB",
                                parent_seeds=(1, 2), indices=(0, 0))
    fails = _parse_practrand_failures(out)
    assert not fails, "Failures interleaving unrelated parents with same idx (fold advancement):\n" + "\n".join(fails)


def test_fold_independent_parents_diff_index():
    # different parents and diff indices
    out = _run_interleaved_fold(n_uint32=1 << 35, tlmax="32GB",
                                parent_seeds=(1, 2), indices=(0, 1))
    fails = _parse_practrand_failures(out)
    assert not fails, "Failures interleaving unrelated parents with different idx (fold advancement):\n" + "\n".join(fails)


def test_split_independence_small():
    # Full BigCrush-size exercise (32GB) using two *distinct* children of the
    # same parent.  indices=(0,1) produces two different siblings via fold_in;
    # using (0,0) would give identical keys and trivially fail.
    out = _run_interleaved(n_uint32=1 << 35, tlmax="32GB",
                           parent_seeds=(0, 0), indices=(0, 1))
    fails = _parse_practrand_failures(out)
    assert not fails, "Failures when interleaving two sibling streams:\n" + "\n".join(fails)


def test_independent_parents_same_index():
    # two different parents but both fold in index 0 (P sequence identical)
    out = _run_interleaved(n_uint32=1 << 35, tlmax="32GB", parent_seeds=(1, 2), indices=(0, 0))
    fails = _parse_practrand_failures(out)
    assert not fails, "Failures interleaving unrelated parents with same idx:\n" + "\n".join(fails)


def test_independent_parents_diff_index():
    # two different parents and different fold-in indices
    out = _run_interleaved(n_uint32=1 << 35, tlmax="32GB", parent_seeds=(1, 2), indices=(0, 1))
    fails = _parse_practrand_failures(out)
    assert not fails, "Failures interleaving unrelated parents with different idx:\n" + "\n".join(fails)
