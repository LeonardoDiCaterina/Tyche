# Tyche

A JAX-compatible PRNG built on iterated quadratic maps over integer matrix rings, designed to map directly onto tensor-core hardware.

## Mathematical foundation

### Algebraic setting

Tyche operates over the matrix ring $M_B(\mathbb{Z}\_{2^{16}})$ — the ring of $B \times B$ matrices with entries in unsigned 16-bit integers. The core mixing primitive is the quadratic map:

$$X \mapsto f(X) = \tau\!\bigl(\,c \cdot (X^2 + W_r) \oplus (c \cdot (X^2 + W_r) \gg 16)\,\bigr)$$

where $X^2 = X \cdot X$ is matrix multiplication over $\mathbb{Z}\_{2^{32}}$, $W_r$ is a key-dependent weight matrix, $c$ is an odd constant (bijection on $\mathbb{Z}\_{2^{32}}$), and $\tau$ truncates to $\mathbb{Z}\_{2^{16}}$.

### Round function

Given a counter block $x_0 \in M_B(\mathbb{Z}\_{2^{16}})$ and key matrices $\{W_r\}_{r=0}^{R-1}$:

$$x_{r+1} = \tau\!\bigl(\,\text{mix}(x_r^2 + W_r)\,\bigr), \qquad r = 0, \ldots, R-1$$

where $\text{mix}(a) = (a \cdot c) \oplus ((a \cdot c) \gg 16)$. The three stages per round are:

| Stage | Operation | Purpose |
|---|---|---|
| **FMA** | $a = x \cdot x + W_r$ | Quadratic nonlinearity via integer matmul carries |
| **Odd multiply** | $a = a \cdot c,\; c = \texttt{0x94D049BB}$ | Full carry cascade — breaks low-bit linearity |
| **XOR fold** | $a = a \oplus (a \gg 16)$ | Folds carry-enriched high bits into low bits |

### Invertibility guarantee

Counter blocks are embedded into $GL_B(\mathbb{Z}\_{2^{16}})$ (invertible matrices) via a triangular embedding: diagonal entries are forced odd, strict upper triangle entries even. This structure guarantees a unit determinant and prevents zero-absorption under squaring, but it does introduce a mild bias in the first round’s inputs (upper‑triangle elements have LSB=0).  Empirically the bias is undetectable after $R=4$ rounds for $B=4$ (measurement requires thousands of samples; formal diffusion bounds remain an open question), and the trade‑off is deemed acceptable for the strong invertibility guarantee.

### Key derivation (split / fold_in)

Child keys are derived by a quadratic perturbation of the weight matrices:

$$W_r' = W_r^2 + P(i, r)$$

where $P(i, r)$ is a round-dependent perturbation matrix obtained by hashing the child index $i$ together with the round number $r$ using a 2-multiply bijective hash:

$$h(x) = \text{xor-shift-multiply-xor-shift-multiply-xor-shift}(x)$$

A fresh perturbation is computed for **each round** (e.g. by mixing the round index into $h$) so that sibling keys derived from the same parent differ unpredictably across rounds.  Earlier versions used a single shared $P(i)$ for all rounds, which created linear relationships between siblings.


### Nonlinearity analysis

- **Integer matmul** provides carry-chain nonlinearity, but low-order bits of $\mathbb{Z}_{2^n}$ multiplication are linear over $GF(2)$.
- **Odd multiply** compensates: multiplication by an odd constant is a bijection on $\mathbb{Z}_{2^{32}}$ that forces carry propagation from LSB to MSB, injecting nonlinearity at every bit position.
- **XOR fold** transports the now-nonlinear high bits back into the low 16 bits before truncation.
- One quadratic layer per round provides nonlinearity; successive rounds compound that effect, giving very rapid growth in complexity even though each round is simple.

> **Note:** we truncate to 16 bits each round, so the algebraic degree of the low-order output bits is much lower than what a full‑precision analysis would suggest. The stated growth mainly applies to the high‑bits that survive the truncation.

## Hardware mapping

### Tensor core path (FMA stage)

The dominant operation $X^2 + W$ is a fused matrix-multiply-accumulate:

```
                ┌─────────────┐
  x (uint16) ──┤  Tensor Core │── acc (uint32)
  x (uint16) ──┤   HMMA/IMMA  │      │
                └─────────────┘      (+) ← W_r (uint32)
                                      │
                                  acc_32
```

- **GPU (NVIDIA):** Maps to `IMMA` (INT8→INT32) or `HMMA` (FP16→FP32) tensor core instructions. (Practical note: on Ampere/Hopper the smallest full INT8 tile is 8×16×32; a 4×4 matmul cannot fill a tile and executes on scalar ALUs or as a partially-utilised tile. Larger $B$ values are required to amortise tensor core overhead.)
- **TPU:** Maps to MXU systolic array; int8→int32 accumulation is native.
- **CPU:** Standard GEMM; benefits from AVX-512 VNNI for int8 paths.

### ALU path (odd multiply + XOR fold)

Post-matmul mixing runs on scalar/vector ALU, overlapping with the next tensor core dispatch:

```
  acc_32 ──[× ODD_MULT]──[⊕ (>> 16)]──[trunc uint16]── x_next
           │              │              │
         1 IMUL        1 SHR + XOR    free (cast)
```

Total ALU cost: **2 integer ops per element per round** — negligible vs. the matmul.

### Memory hierarchy

| Data | Size ($B=4$, $R=4$) | Residency |
|---|---|---|
| State $x$ | $4 \times 4 \times 2\text{B} = 32\text{B}$ | Registers |
| Weight matrices $W_r$ | $4 \times 4 \times 4 \times 4\text{B} = 256\text{B}$ | Shared memory / L1 |
| Counter block | $32\text{B}$ | Registers |

Everything fits in on-chip storage. **Zero global memory traffic** during the round loop.

### Pallas / Triton lowering

The round loop is a fixed-iteration `for` unrolled at trace time. Each iteration is:
1. `tl.dot(x, x)` — tensor core
2. `+ W_r` — ALU add
3. `* ODD_MULT` — ALU multiply
4. `^ (>> 16)` — ALU shift + XOR
5. `.to(uint16)` — free cast

No control flow, no memory loads inside the loop, no synchronisation barriers. A single Pallas `pallas_call` with `grid=(N,)` maps one counter block per program, fully saturating the GPU with independent work.

## Testing & validation

Tyche ships with a comprehensive test suite covering correctness, JAX compatibility, statistical quality, and external randomness testing. All tests below pass on the current codebase (macOS ARM / Python 3.12 / JAX with x64 enabled).

### Unit tests (108 passed)

Run with `pytest tests/ -m "not crush"`:

| Module | Tests | What it covers |
|---|---|---|
| `test_smoke.py` | 5 | Impl registration, key creation, split, uniform |
| `test_seed.py` | 22 | Key shape, dtype, determinism, edge cases across 6 seeds |
| `test_split.py` | 10 | Output shape, child distinctness, determinism, parent ≠ child |
| `test_fold_in.py` | 10 | Shape, determinism, sensitivity to key/data, edge-case data |
| `test_random_bits.py` | 11 | Shape, dtype, determinism, key-sensitivity, child-bit diversity |

### JAX compatibility (23 passed)

| Module | Tests | What it covers |
|---|---|---|
| `test_against_builtin.py` | 10 | Shape/range/permutation parity with Threefry for `uniform`, `normal`, `randint`, `shuffle` |
| `test_jit.py` | 7 | JIT compilation of `key`, `split`, `fold_in`, `random_bits`, `uniform`, `normal`; determinism under re-JIT |
| `test_vmap.py` | 6 | `vmap` over `split`, `fold_in`, `normal`; distinct samples per key; `vmap` ∘ `jit` composition |

### Statistical tests (27 passed)

| Module | Tests | What it covers |
|---|---|---|
| `test_avalanche.py` | 3 | Seed / fold-in / split avalanche — >30 % bit flips per input change |
| `test_bits.py` | 6 | Bit balance, byte distribution, run-length check, MSB/LSB balance, seed diversity |
| `test_indipendence.py` | 6 | Serial & higher-lag autocorrelation, split/fold-in independence, runs test, 2-D uniformity |
| `test_uniformity.py` | 11 | KS & χ² goodness-of-fit for `uniform`; mean/variance checks; cross-key uniformity; `normal` distribution test |

### PractRand external randomness tests

PractRand 0.95 is used as a black-box stream tester. The generator pipes raw `uint32` output from `jax.random.bits` (Tyche impl) into PractRand's `RNG_test stdin32`.

| Level | Data size | Result | Command |
|---|---|---|---|
| **SmallCrush** | 128 MB | **PASS** — 156 tests, 0 anomalies | `pytest tests/crush/test_smallcrush.py -m crush` |
| **Crush** | 1 GB | **PASS** — 0 anomalies | `pytest tests/crush/test_crush.py -m crush` |
| **BigCrush** | 32 GB | Not yet run (hours-long) | `pytest tests/crush/test_bigcrush.py -m crush` |

> **Note:** PractRand tests require the `RNG_test` binary. On macOS ARM, the bundled source in `tests/crush/bridge/PractRand/` includes a POSIX `read()` fix for stdin binary mode. Compile with `make -C tests/crush/bridge`.

### Notebook tests (proof-of-concept)

The `tyche_poc.ipynb` notebook includes additional interactive validation:

- **Strict Avalanche Criterion (SAC):** embed-then-flip methodology over 1000 samples; round-count sweep from R=1 to R=8
- **Per-bit frequency test:** all 16 bit positions within ±0.02 of 0.5 ideal
- **Low-bit serial correlation:** χ² test on consecutive-pair distribution (p > 0.001 for all bit positions)
- **Linear complexity (Berlekamp-Massey):** LC/N > 0.3 for 2048-bit streams at every bit position — confirms nonlinearity over GF(2)

## Usage

```python
import jax
from tyche import impl as tyche_impl

key = jax.random.key(42, impl=tyche_impl)
samples = jax.random.normal(key, shape=(1_000_000,))
```

## Configuration

```python
from tyche import TycheConfig

cfg = TycheConfig(block_size=4, num_rounds=8)
impl = cfg.build()
key = jax.random.key(0, impl=impl)
```

| Parameter | Default | Effect |
|---|---|---|
| `block_size` | 4 | Matrix dimension $B$; output per block = $B^2$ uint16 |
| `num_rounds` | 4 | Number of quadratic-map iterations $R$ |
