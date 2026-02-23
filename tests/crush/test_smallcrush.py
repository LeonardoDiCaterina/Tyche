"""
tests/crush/test_smallcrush.py

PractRand quick test — 128 MB, runs in seconds.
Run: pytest tests/crush/test_smallcrush.py -v -m crush
"""
import pytest
from crush.practrand_utils import _run_practrand, _parse_practrand_failures

pytestmark = pytest.mark.crush


def test_practrand_128mb():
    """Tyche must pass PractRand at 128 MB with no failures."""
    # 128 MB = 2^27 bytes = 33_554_432 uint32 values
    # Generate 2x to give PractRand headroom for folded transforms
    output = _run_practrand(n_uint32=67_108_864, tlmax="128MB")
    failures = _parse_practrand_failures(output)
    assert len(failures) == 0, (
        "PractRand failures at 128MB:\n" + "\n".join(f"  - {f}" for f in failures)
    )