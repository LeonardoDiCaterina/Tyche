"""
tests/crush/test_smallcrush.py

PractRand quick test — now upgraded to 32GB to provide maximal coverage
instead of the original 128MB run.  Use with care; it will take hours.
Run: pytest tests/crush/test_smallcrush.py -v -m crush
"""
import pytest
from crush.practrand_utils import _run_practrand, _parse_practrand_failures

pytestmark = pytest.mark.crush


def test_practrand_128mb():
    """Tyche must pass PractRand at 128 MB with no failures."""
    # 32 GB = 2^35 bytes = 8_589_934_592 uint32 values
    # Generate 2x to give PractRand headroom for folded transforms
    output = _run_practrand(n_uint32=17_179_869_184, tlmax="32GB")
    failures = _parse_practrand_failures(output)
    assert len(failures) == 0, (
        "PractRand failures at 128MB:\n" + "\n".join(f"  - {f}" for f in failures)
    )