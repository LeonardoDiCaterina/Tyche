"""
tests/crush/test_crush.py

PractRand medium test — 1GB, runs in ~minutes.
Run: pytest tests/crush/test_crush.py -v -m crush
"""
import pytest
from crush.practrand_utils import _run_practrand, _parse_practrand_failures

pytestmark = pytest.mark.crush

def test_practrand_1gb():
    output = _run_practrand(n_uint32=250_000_000, tlmax="1GB")
    failures = _parse_practrand_failures(output)
    assert len(failures) == 0, (
        "PractRand failures at 1GB:\n" + "\n".join(f"  - {f}" for f in failures)
    )
