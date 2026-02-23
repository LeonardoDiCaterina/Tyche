"""Shared PractRand utilities for crush test files.

Requires ``RNG_test`` (PractRand >= 0.95) on PATH.
On macOS ARM the bundled PractRand source must be compiled with the
POSIX read() stdin fix (see tests/crush/bridge/PractRand/tools/).
"""

import shutil
import subprocess
import sys
from pathlib import Path

STREAM = Path(__file__).parent / "stream.py"


def _require_practrand():
    if shutil.which("RNG_test") is None:
        import pytest
        pytest.skip("RNG_test not on PATH. See tests/crush/README.md for setup.")


def _run_practrand(n_uint32: int, tlmax: str = "128MB") -> str:
    """Generate Tyche stream and feed it to PractRand.

    Pipes the stream process stdout directly into PractRand stdin.
    Requires the bundled PractRand build with the POSIX read() fix
    (see tests/crush/bridge/PractRand/tools/RNG_from_name.h).
    """
    _require_practrand()

    stream_proc = subprocess.Popen(
        [sys.executable, str(STREAM), "--n", str(n_uint32)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    practrand_proc = subprocess.Popen(
        ["RNG_test", "stdin32", "-tlmax", tlmax],
        stdin=stream_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Allow stream_proc to receive SIGPIPE if PractRand exits early
    stream_proc.stdout.close()
    stdout, stderr = practrand_proc.communicate()
    stream_proc.wait()
    return stdout.decode() + stderr.decode()


def _parse_practrand_failures(output: str) -> list:
    return [
        line.strip() for line in output.splitlines()
        if "FAIL" in line.upper() or "VERY SUSPICIOUS" in line.upper()
    ]