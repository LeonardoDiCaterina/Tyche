#!/usr/bin/env bash
# Run environment setup, tests, and benchmarks for Tyche on a fresh machine.
# Usage: bash run_on_new_machine.sh       # default: venv + run tests + run benchmarks
#        SKIP_BENCHMARKS=1 bash run_on_new_machine.sh

set -euo pipefail
cd "$(dirname "$0")" || exit 1
ROOT=$(pwd)
VENV_DIR="$ROOT/.venv"
PYTHON=${PYTHON:-python3}
SKIP_BENCHMARKS=${SKIP_BENCHMARKS:-0}

echo "== Tyche setup script =="
echo "Repository: $ROOT"

command_exists() { command -v "$1" >/dev/null 2>&1; }

# 1) Python sanity check
if ! command_exists "$PYTHON"; then
  echo "ERROR: $PYTHON not found. Install Python 3.10+ and retry." >&2
  exit 2
fi
PYVER=$($PYTHON -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')
if [[ $(echo "$PYVER >= 3.10" | bc -l) -ne 1 ]]; then
  echo "ERROR: Python >= 3.10 required (found $PYVER)." >&2
  exit 2
fi

# 2) Create + activate venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtualenv at $VENV_DIR..."
  $PYTHON -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# 3) Upgrade packaging tools
echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# 4) Install project + dev dependencies
echo "Installing Tyche (editable) and dev dependencies..."
set +e
python -m pip install -e '.[dev]' >/dev/null 2>&1
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "Fallback: installing package and pytest+pytest-benchmark explicitly..."
  python -m pip install -e . pytest pytest-benchmark
fi

# 5) Optional: detect GPU and print guidance (do not auto-install GPU jaxlib)
if command_exists nvidia-smi || [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "\nGPU detected — if you want GPU-accelerated JAX, install a matching jaxlib wheel." 
  echo "See: https://github.com/google/jax#pip-installation"
else
  echo "No CUDA GPU detected — CPU JAX will be used."
fi

# 6) Run tests
echo "\nRunning test suite (this may take a minute)..."
pytest -q

# 7) Run benchmarks (unless skipped)
if [[ "$SKIP_BENCHMARKS" -eq 1 ]]; then
  echo "Skipping benchmarks as requested (SKIP_BENCHMARKS=1)."
else
  echo "\nRunning throughput benchmarks and saving results to benchmarks/bench_throughput_results.txt..."
  mkdir -p benchmarks
  pytest benchmarks/test_bench_throughput.py -q > benchmarks/bench_throughput_results.txt 2>&1 || true
  echo "Benchmarks complete — tail of results:" 
  tail -n 40 benchmarks/bench_throughput_results.txt || true
fi

# 8) Done
echo "\nSetup + tests complete."
echo "- Virtualenv: $VENV_DIR (activate with: source $VENV_DIR/bin/activate)"
echo "- Benchmark results: benchmarks/bench_throughput_results.txt"

echo "If you want GPU JAX, follow instructions at https://github.com/google/jax#pip-installation"
exit 0
