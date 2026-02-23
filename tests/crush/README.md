# Statistical Crush Tests (PractRand)

This directory contains scaffolding to run Tyche through PractRand —
an industry-standard empirical PRNG test suite.

## Platform support

| Platform        | Status  | Notes                                      |
|-----------------|---------|--------------------------------------------|
| Linux x86_64    | ✅      | Works natively                             |
| macOS Intel     | ✅      | Works natively                             |
| macOS ARM (M1+) | ⚠️      | stdin binary mode bug in PractRand pre-0.95|

## Setup

### Linux / macOS Intel
```bash
curl -L -o PractRand.zip https://sourceforge.net/projects/pracrand/files/PractRand-pre0.95.zip/download
unzip PractRand.zip -d PractRand && cd PractRand
g++ -std=c++14 -O3 -o RNG_test \
    tools/RNG_test.cpp \
    src/*.cpp src/RNGs/*.cpp src/RNGs/other/*.cpp \
    -Iinclude -pthread
sudo mv RNG_test /usr/local/bin/
```

### macOS ARM (M1/M2/M3) — use Docker
PractRand's stdin binary mode is broken on Apple Silicon. Use Docker:

```bash
# Build once
docker build -t practrand-linux tests/crush/docker/

# Run tests
docker run --rm \
    -v $(pwd):/tyche \
    practrand-linux \
    bash -c "cd /tyche && python tests/crush/stream.py --n 25000000 2>/dev/null | RNG_test stdin32 -tlmax 100MB"
```

## Running the tests

```bash
# Quick check (100MB, ~seconds)
pytest tests/crush/test_smallcrush.py -v -m crush

# Medium (1GB, ~minutes)  
pytest tests/crush/test_crush.py -v -m crush

# Full certification (32GB, ~hours)
pytest tests/crush/test_bigcrush.py -v -m crush
```

## Interpreting PractRand results

PractRand reports progressively at 64MB, 128MB, 256MB, etc.
- `no anomalies` — all good
- `unusual` — worth noting, not a failure  
- `suspicious` — investigate
- `FAIL` — genuine flaw in the algorithm

One FAIL on a single run is not always fatal — re-run to confirm.
Consistent FAILs across runs indicate a real problem.