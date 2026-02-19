"""
tests/crush/test_smallcrush.py

PractRand quick test — 100MB, runs in seconds.

NOTE: PractRand stdin is broken on macOS ARM (M1/M2/M3).
Use Docker on Apple Silicon — see tests/crush/README.md.

Run:
    pytest tests/crush/test_smallcrush.py -v -m crush
Or directly:
    python tests/crush/stream.py --n 25000000 2>/dev/null | RNG_test stdin32 -tlmax 100MB
"""

import platform
import shutil
import subprocess
import sys
import pytest
from pathlib import Path

pytestmark = pytest.mark.crush

STREAM = Path(__file__).parent / "stream.py"


def _require_practrand():
    if shutil.which("RNG_test") is None:
        pytest.skip("RNG_test not on PATH. See tests/crush/README.md for setup.")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        pytest.skip(
            "PractRand stdin is broken on macOS ARM. "
            "Use Docker instead — see tests/crush/README.md."
        )


def _run_practrand(n_uint32: int, tlmax: str = "100MB") -> str:
    _require_practrand()
    stream_proc = subprocess.Popen(
        [sys.executable, str(STREAM), "--n", str(n_uint32)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(
        ["RNG_test", "stdin32", "-tlmax", tlmax, "-tlfail"],
        stdin=stream_proc.stdout,
        capture_output=True,
        text=True,
    )
    stream_proc.wait()
    return result.stdout + result.stderr


def _parse_practrand_failures(output: str) -> list:
    return [
        line.strip() for line in output.splitlines()
        if "FAIL" in line.upper() or "VERY SUSPICIOUS" in line.upper()
    ]


def test_practrand_100mb():
    """Tyche must pass PractRand at 100MB with no failures."""
    output = _run_practrand(n_uint32=25_000_000, tlmax="100MB")
    failures = _parse_practrand_failures(output)
    assert len(failures) == 0, (
        "PractRand failures at 100MB:\n" + "\n".join(f"  - {f}" for f in failures)
    )