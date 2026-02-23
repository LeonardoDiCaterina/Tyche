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

Counter blocks are embedded into $GL_B(\mathbb{Z}\_{2^{16}})$ (invertible matrices) via a triangular embedding: diagonal entries are forced odd, strict upper triangle entries even. Since $\det(X)$ is then odd (product of diagonal), $X$ is a unit in $M_B(\mathbb{Z}_{2^{16}})$, preventing degenerate zero-absorption under squaring.

### Key derivation (split / fold_in)

Child keys are derived by a quadratic perturbation of the weight matrices:

$$W_r' = W_r^2 + P(i)$$

where $P(i)$ is a perturbation matrix expanded from the child index $i$ via a 2-multiply bijective hash:

$$h(x) = \text{xor-shift-multiply-xor-shift-multiply-xor-shift}(x)$$

This is branch-free, has no sequential dependency, and maps directly to a GPU thread ID.

### Nonlinearity analysis

- **Integer matmul** provides carry-chain nonlinearity, but low-order bits of $\mathbb{Z}_{2^n}$ multiplication are linear over $GF(2)$.
- **Odd multiply** compensates: multiplication by an odd constant is a bijection on $\mathbb{Z}_{2^{32}}$ that forces carry propagation from LSB to MSB, injecting nonlinearity at every bit position.
- **XOR fold** transports the now-nonlinear high bits back into the low 16 bits before truncation.
- Per-round algebraic degree: $\deg \approx 2$ (from squaring), compounding to $2^R$ over $R$ rounds.

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

- **GPU (NVIDIA):** Maps to `IMMA` (INT8→INT32) or `HMMA` (FP16→FP32) tensor core instructions. For $B=4$, a single warp-level MMA fills one tensor core tile.
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
