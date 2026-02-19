"""Shared PractRand utilities for crush test files."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

STREAM = Path(__file__).parent / "stream.py"


def _require_practrand():
    if shutil.which("RNG_test") is None:
        import pytest
        pytest.skip("RNG_test not on PATH. See tests/crush/README.md for setup.")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import pytest
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