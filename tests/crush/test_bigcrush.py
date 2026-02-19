"""
tests/crush/test_bigcrush.py

PractRand exhaustive test — 32GB, runs in ~hours.
Run: pytest tests/crush/test_bigcrush.py -v -m crush
"""
import pytest
from crush.practrand_utils import _run_practrand, _parse_practrand_failures

pytestmark = pytest.mark.crush

def test_practrand_32gb():
    output = _run_practrand(n_uint32=8_000_000_000, tlmax="32GB")
    failures = _parse_practrand_failures(output)
    assert len(failures) == 0, (
        "PractRand failures at 32GB:\n" + "\n".join(f"  - {f}" for f in failures)
    )

def test_practrand_32gb_double_run():
    pytest.skip("Remove skip for final certification.")
