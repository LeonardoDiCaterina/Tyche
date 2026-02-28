"""
tests/crush/test_crush.py

PractRand medium test — now upgraded to 32GB to match BigCrush.
(This file previously exercised 1GB; the dedicated bigcrush module still exists.)
Run: pytest tests/crush/test_crush.py -v -m crush
"""
import pytest
from crush.practrand_utils import _run_practrand, _parse_practrand_failures

pytestmark = pytest.mark.crush

def test_practrand_32gb():
    # 32 GB = 2^35 bytes = 8_589_934_592 uint32 values
    # Generate 2x to give PractRand headroom for folded transforms
    output = _run_practrand(n_uint32=17_179_869_184, tlmax="32GB")
    failures = _parse_practrand_failures(output)
    assert len(failures) == 0, (
        "PractRand failures at 32GB:\n" + "\n".join(f"  - {f}" for f in failures)
    )
