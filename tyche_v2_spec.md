# Tyche v2 — Design Specification

## Motivation

Tyche v1 operated over K independent (B×B) uint16 counter blocks.
This design created a fundamental mismatch with tensor core hardware:
each block was too small to fill a tile, so the matmul either ran on
scalar ALUs or wasted most of a partially-utilised tile.

v2 eliminates this mismatch by treating the **entire batch as one native
int8 tile**. There are no casts, no reinterpretations, and no memory
traffic between stages. The single cast happens once, at output, after
all rounds are complete.

---

## Core Principles

1. **Native int8 throughout the round loop.** No uint16 state. The
   tensor core ingests int8 and produces int32; the ALU stage consumes
   int32 and truncates back to int8. That is the only dtype transition
   inside the hot path.

2. **Counter is positional, not structural.** The tile index `n` encodes
   which batch you are on. The element position `(i, j)` within the tile
   encodes which sample within the batch. No field inside the matrix
   carries counter information — it is fully implicit.

3. **One IMMA call per round.** The entire batch is one `(T × T)` int8
   matmul, properly sized to fill a tensor core tile.

4. **Key derivation is unchanged.** Split and fold_in operate exclusively
   on weight matrices `W_r`. The counter axis and the key axis are
   orthogonal by construction.

5. **Single output cast.** After the final round the `(T × T)` int8
   tile is reinterpreted as uint32 or uint64 exactly once, at the call
   site of `random_bits`. No intermediate materialisation.

---

## Mathematical Foundation

### Algebraic Setting

Tyche v2 operates over the matrix ring $M_T(\mathbb{Z}\_{2^8})$ — the ring of $T \times T$ matrices with entries in signed 8-bit integers. Each tile is a single element of this ring. The round function is a quadratic map composed with an element-wise bijection:

$$X \mapsto f(X) = \tau\!\Bigl(c \cdot \bigl(\tilde{X}^2 + W_r\bigr) \oplus \bigl(c \cdot (\tilde{X}^2 + W_r) \gg 16\bigr)\Bigr)$$

where:
- $\tilde{X} \in M_T(\mathbb{Z}\_{2^{32}})$ is the zero-extension of $X$ to 32-bit entries
- $\tilde{X}^2 = \tilde{X} \cdot \tilde{X}$ is matrix multiplication with 32-bit accumulation
- $W_r \in M_T(\mathbb{Z}\_{2^{32}})$ is a key-dependent weight matrix for round $r$
- $c = \texttt{0x94D049BB}$ is an odd constant, a bijection on $\mathbb{Z}\_{2^{32}}$ (element-wise)
- $\gg 16$ is an arithmetic right-shift by 16 bits (element-wise)
- $\tau$ truncates each 32-bit element to its low 8 bits, mapping back into $M_T(\mathbb{Z}\_{2^8})$

The full $R$-round map starting from initial tile $X_0(n)$ is:

$$X_{r+1} = f_r(X_r) = \tau\!\Bigl(c \cdot (\tilde{X}_r^2 + W_r) \oplus (c \cdot (\tilde{X}_r^2 + W_r) \gg 16)\Bigr), \qquad r = 0, \ldots, R-1$$

### Round Structure

Each round decomposes into three distinct stages with different algebraic roles:

| Stage | Operation | Domain | Purpose |
|---|---|---|---|
| **FMA** | $A = \tilde{X}^2 + W_r$ | $M_T(\mathbb{Z}\_{2^{32}})$ | Quadratic nonlinearity via carry-chain integer matmul |
| **Odd multiply** | $A \leftarrow A \cdot c$ | $M_T(\mathbb{Z}\_{2^{32}})$ (element-wise) | Full carry cascade LSB→MSB; bijection on $\mathbb{Z}\_{2^{32}}$ |
| **XOR fold + truncate** | $\tau(A \oplus (A \gg 16))$ | $M_T(\mathbb{Z}\_{2^8})$ | Folds carry-enriched high bits into low 8 bits |

### Nonlinearity Analysis

- **Integer matmul** provides carry-chain nonlinearity across matrix entries. Each output element $A_{ij} = \sum_k X_{ik} X_{kj} + W_{r,ij}$ accumulates $T$ cross-products, meaning every input entry influences every output entry — full intra-tile diffusion in one round.
- **Odd multiply** compensates for low-bit GF(2) linearity of integer multiplication. For odd $c$, the map $a \mapsto a \cdot c$ is a bijection on $\mathbb{Z}\_{2^{32}}$ that forces a carry chain from LSB to MSB, injecting nonlinearity at every bit position.
- **XOR fold** transports the carry-enriched high 16 bits into the low bits before truncation, ensuring that the retained 8 bits inherit the nonlinearity from the full 32-bit computation rather than only the low-order linear bits.
- **Truncation to 8 bits** loses the high bits each round, resetting algebraic degree. The statistical strength comes from empirical diffusion rather than from provable degree bounds — successive rounds compound the carry-chain mixing, and the matmul ensures every element participates in every output.

### Counter Embedding

The initial tile $X_0$ for batch index $n$ is constructed as:

$$X_0(n)[i,j] = \tau_8\!\bigl(h(k_{\text{mix}} \oplus n \cdot M_1 \oplus i \cdot M_2 \oplus j \cdot M_3)\bigr)$$

where $k_{\text{mix}} = c \cdot \bigoplus_\ell k_\ell$ is derived by XOR-folding all key words then multiplying by Knuth's constant, $h$ is the 2-multiply bijective hash (fast\_mix), and $M_1, M_2, M_3$ are distinct odd constants. The embedding mode is a configurable parameter that controls the structure of $X_0$ — see the Tile Embedding section.

### Key Derivation

Child keys are derived by a quadratic perturbation of the weight matrices. For child index $v$:

$$W_r' = W_r^2 + P\!\bigl(h(v) \oplus r \cdot M_\phi,\; T\bigr)$$

where $P(\cdot, T)$ expands a scalar to a $(T \times T)$ matrix via element-wise fast\_mix seeded by position, and $M_\phi = \texttt{0x9E3779B9}$ is the golden-ratio constant used as a round salt. The per-round salt ensures that siblings $v$ and $v'$ differ across all rounds simultaneously, not just in the first perturbation.

---

## Hardware Mapping

### FMA Stage — Tensor Core Path

The dominant operation $\tilde{X}^2 + W_r$ is a fused matrix-multiply-accumulate executed as a single IMMA (Integer Matrix Multiply-Accumulate) instruction:

```
  X (int8, T×T) ──┐
                   ├──[ Tensor Core IMMA ]──► acc (int32, T×T)
  X (int8, T×T) ──┘         │
                              └──(+) W_r (int32, T×T)  ← shared memory
                                   │
                               acc_32 (int32, T×T)
```

- **GPU (NVIDIA Ampere/Hopper):** maps to `IMMA` (m16n16k32 or larger). With $T = 16$, the tile exactly fills one warp-level IMMA instruction. No padding, no partial tile waste.
- **TPU:** maps to MXU systolic array; int8→int32 accumulation is the native path. XLA fuses the vmap over tile indices into a single MXU dispatch automatically.
- **CPU:** GEMM via AVX-512 VNNI (int8 dot-product instructions); no tensor core benefit but the matmul still vectorises well.

### ALU Stage — Odd Multiply and XOR Fold

The post-matmul mixing runs on scalar/vector ALU, overlapping with the next IMMA dispatch:

```
  acc_32 (int32, T×T)
       │
       ├──[× 0x94D049BB]──► carry-cascaded int32   (1 IMUL per element)
       │
       ├──[⊕ (>> 16)]─────► high bits folded down  (1 SHR + 1 XOR per element)
       │
       └──[trunc int8]─────► X_next (int8, T×T)    (free cast, no instruction)
```

Total ALU cost: **2 integer ops per element per round** — negligible versus the matmul. On GPU these ops pipeline in the shadow of the next IMMA.

### Memory Hierarchy

| Data | Size (T=16, R=4) | Residency |
|---|---|---|
| State $X$ | $16 \times 16 \times 1\text{B} = 256\text{B}$ | Registers |
| Weight matrices $W_r$ | $4 \times 16 \times 16 \times 4\text{B} = 4\text{KB}$ | Shared memory / L1 |
| Tile embedding $X_0$ | $256\text{B}$ computed once | Registers (scalar ALU pre-loop) |

Everything fits on-chip. **Zero global memory traffic** during the round loop.

### Pallas / Triton Lowering

The round loop is a fixed-iteration `for` unrolled at trace time. Each iteration:

1. `tl.dot(x, x)` — IMMA, single tile, fully occupied
2. `+ W_r` — ALU broadcast add (W_r is a compile-time constant per round)
3. `* ODD_MULT` — scalar IMUL broadcast
4. `^ (>> 16)` — scalar SHR + XOR
5. `.to(int8)` — free cast

No control flow, no memory loads inside the loop, no synchronisation barriers. A `pallas_call` with `grid=(N_tiles,)` assigns one tile per program, saturating the GPU with independent work.

---

## Competitive Positioning

### When Tyche v2 Has an Advantage

Tyche v2 is not a general-purpose PRNG. It targets a specific regime where existing designs leave compute on the table.

**1. Large-batch generation on tensor-core hardware**

Philox and Threefry were designed around scalar multiply and add-rotate-XOR respectively — primitives that are universally available but do not use tensor cores at all. On an A100 or H100, Philox-4×32 runs entirely on CUDA integer ALUs, using none of the tensor core capacity that dominates the chip's FLOP budget. Tyche v2's IMMA call routes through the part of the chip with 10–50× the throughput of scalar ALUs.

This advantage is only realised when generating large batches — the tile granularity is $T^2 / 4$ uint32 values per call (64 for T=16). For ML training workloads that call `jax.random.normal(key, shape=(batch, dim))` with batch × dim in the thousands to millions, this is always satisfied.

**2. TPU workloads**

The MXU systolic array on TPU is always a batched matmul. There is no scalar ALU path of comparable throughput. Philox and Threefry do not benefit from the MXU at all; their inner loops are sequences of integer ops that the TPU executes sub-optimally. Tyche v2's matmul maps directly onto the MXU's native operation.

**3. Monte Carlo kernels in JAX where generation is inside a `jit` + `vmap`**

When random generation is fused with computation (e.g., dropout masks, stochastic depth, MC integration), XLA can schedule the Tyche IMMA alongside compute-heavy matmuls from the model, filling tensor core utilisation gaps. Scalar PRNGs cannot participate in this scheduling.

### When Tyche v2 Does Not Have an Advantage

| Scenario | Better choice | Reason |
|---|---|---|
| Single-sample or small-batch generation | Philox-4×32 | Tile overhead not amortised; Philox has near-zero granularity cost |
| CPU-only deployment | Threefry-4×64 | AVX-512 VNNI helps but matmul cost dominates; Threefry is simpler and equally fast |
| Cryptographic use | AES-CTR | Tyche makes no cryptographic claims |
| Memory-constrained environments | Philox | 4 KB key vs 24 bytes for Philox |
| Fine-grained random access (arbitrary counter) | Any counter-based PRNG | Tyche's tile structure makes random access to sample $k$ slightly more complex |

### Comparison With Salmon et al. (2011)

The Salmon et al. paper established the counter-based PRNG paradigm that JAX's Threefry and Philox implement. Tyche v2 sits firmly within this paradigm — it is a keyed bijection applied to a counter, with the counter encoded positionally as tile index × tile position.

The key conceptual difference is the choice of hardware primitive:

| Design | Core primitive | Hardware target era |
|---|---|---|
| Philox | mulhi/mullo (word-wide multiply) | Scalar integer units, 2011 GPUs |
| Threefry | Add-Rotate-XOR | Any ALU, portable |
| Tyche v2 | INT8 matmul (IMMA) | Tensor cores, 2020+ GPUs and TPUs |

Salmon et al. note that "one of the PRNGs we introduce is the fastest we know of on GPUs" — referring to Philox at 202 GB/s on a GTX 580. An H100 tensor core can sustain >3 TB/s of INT8 compute. The design space has changed by more than an order of magnitude, and Tyche v2 is the first counter-based PRNG designed to exploit it.

---

## Parameters

```python
@dataclass
class TycheConfig:
    tile_size:  int = 16       # T — must be tensor-core-friendly: 16, 32, 64
    num_rounds: int = 4        # R
    embedding:  str = "hash"   # how tile index n is embedded into X_0
```

`tile_size` replaces `block_size`. Valid values are 16, 32, and 64 — the
minimum sizes that fill an INT8 IMMA tile on Ampere/Hopper (m16n16k32
effective shape after padding). T=16 is the recommended default.

`embedding` is a first-class experimental axis. See the Embedding section.

---

## Data Layout

| Tensor | Shape | Dtype | Residency |
|---|---|---|---|
| State `X` | `(T, T)` | int8 | Registers |
| Weight matrices `W_r` | `(R, T, T)` | int32 | Shared memory / L1 |
| Accumulator `acc` | `(T, T)` | int32 | Registers (internal to TC) |

Key size in uint32 words: `R × T × T`.

For the default config (T=16, R=4): key = 1024 uint32 words = 4 KB.
For T=32, R=4: key = 4096 uint32 words = 16 KB — fits in L1 on all targets.

---

## Round Loop

```
for r in 0..R-1:

  Stage 1 — FMA  (Tensor Core)
    acc = X_int8 @ X_int8 + W_r        shape: (T,T) int32
    — single IMMA call, tile fully occupied
    — W_r added as int32 accumulator bias (free in the MAC pipeline)

  Stage 2 — Odd multiply  (ALU, element-wise)
    acc = acc * ODD_MULT                ODD_MULT = 0x94D049BB (uint32)
    — bijection on Z_{2^32}
    — forces full carry cascade LSB → MSB
    — breaks GF(2) linearity at every bit position

  Stage 3 — XOR fold  (ALU, element-wise)
    acc = acc ^ (acc >> 16)
    — folds carry-enriched high 16 bits into low 16 bits
    — no memory traffic

  Stage 4 — Truncate  (free cast)
    X = acc.to(int8)                    low 8 bits, no instruction cost
    — state is back to (T, T) int8 for the next round
```

Total ALU cost per element per round: **2 integer ops** (IMUL + SHR+XOR).
The matmul dominates. ALU overlaps with the next tile dispatch.

---

## Tile Embedding (X_0 construction)

The initial state for tile `n` is a pure function of `n`, the key
mixing constant, and the position `(i, j)`:

```
X_0[i, j] = int8( hash(key_mix, n, i, j) )
```

This is computed once on scalar ALU before the round loop. It has no
memory reads and no control flow. The embedding mode controls the hash
structure and is a configurable parameter to study diffusion rate vs
round count.

### Built-in embedding modes

| Mode | Formula | Notes |
|---|---|---|
| `"hash"` | `int8(fast_mix(key_mix ^ n*M1 ^ i*M2 ^ j*M3))` | Default. Full diffusion from round 1. |
| `"diagonal"` | `int8(fast_mix(key_mix ^ n)) if i==j else 0` | Minimal. Slow diffusion, useful as stress test for R sweep. |
| `"row"` | `int8(fast_mix(key_mix ^ n ^ i*M2))` | Rows differ, columns identical — intermediate diffusion. |
| `"rank1"` | `int8(fast_mix(key_mix^n^i) * fast_mix(key_mix^n^j))` | Outer product structure. Interesting algebraic properties. |

All modes use `fast_mix` (the existing 2-multiply bijective hash) and
the existing `key_mix` constant derived from XOR-folding the full key.

The embedding mode is a **scientific parameter** — sweeping
`(embedding, num_rounds)` with the existing avalanche and uniformity
test suite directly measures how many rounds each mode needs to reach
indistinguishability. This replaces the informal "undetectable after R=4"
claim in v1 with a measurable characterisation.

---

## Counter Structure

```
Tile index n   — which batch (incremented in _random_bits)
Position (i,j) — which sample within the batch

Sample index = n × T² + i × T + j
```

No counter bits are stored inside the matrix. The counter is implicit
in the call sequence. This is identical in semantics to v1 (counter-mode
PRNG) but the counter is positional rather than structural.

---

## Key Derivation — Unchanged

Split and fold_in are structurally identical to v1. The only change is
that `W_r` matrices are now `(T, T)` int32 instead of `(B, B)` uint32.

```
child_key:  W_r' = W_r² + P(hashed_value, r)    for each round r
```

`P` is derived via `_expand_scalar_to_matrix` (existing fast_mix based
hash), extended to produce a `(T, T)` matrix. The round-salt XOR that
fixed the v1 sibling-correlation bug is retained unchanged.

Key independence across calls is fully preserved: different tile indices
produce different X_0 via the embedding; different keys produce different
W_r; the two axes never interact.

---

## Output

After the final round, the `(T, T)` int8 tile holds T² random bytes.
These are reinterpreted as uint32 or uint64 **once**, at the `random_bits`
call site:

```python
flat_i8 = tile.reshape(-1)            # T² int8 values
output   = flat_i8.view(jnp.uint32)   # T²/4 uint32 values
```

For shapes requiring more than T²/4 uint32 values, `_random_bits`
iterates over tile indices `n = 0, 1, 2, ...` and concatenates.
Each tile is fully independent (different X_0 via embedding).

---

## What Is Removed vs V1

| V1 concept | V2 status | Reason |
|---|---|---|
| `tyche_embed` triangular structure | **Removed** | Bias-introducing, no longer needed — embedding replaces it |
| uint16 state | **Removed** | Replaced by native int8 |
| K independent (B×B) blocks | **Replaced** | One (T×T) tile per batch |
| `make_counter_blocks` | **Replaced** | `make_tile(n, key_mix, T, embedding)` |
| `block_size` parameter | **Renamed** | `tile_size`, with tighter valid set {16, 32, 64} |
| `_hash_block` + vmap | **Replaced** | `_hash_tile` operating on full (T×T) tile |
| Mid-loop uint16 truncation | **Removed** | int8 truncation only, at end of each round |

---

## What Is Preserved vs V1

| V1 concept | V2 status |
|---|---|
| `derive_child_key` structure | Unchanged |
| `_fast_mix_u32` / round-salt XOR | Unchanged |
| `expand_seed_to_key` (SplitMix64) | Unchanged |
| `_key_to_matrices` / `_matrices_to_key` | Unchanged (shape changes) |
| ALU stage (odd multiply + XOR fold) | Unchanged |
| Backend protocol | Unchanged (`hash_parallel` signature same) |
| JAX PRNGImpl registration | Unchanged |

---

## File-level Change Summary

| File | Change |
|---|---|
| `algorithm.py` | Remove `tyche_embed`. Replace `make_counter_blocks` + `_hash_block` with `make_tile` + `_hash_tile`. Round loop now int8→int32→int8. |
| `config.py` | `block_size` → `tile_size`. Add `embedding` param. Update `_random_bits` to iterate tile indices. Key size formula unchanged in structure. |
| `backend.py` | `hash_parallel` signature unchanged. `make_counter_blocks` → `make_tiles`. |
| `backend_jax.py` | Route to new `_hash_tile` / `make_tile`. vmap over tile indices. |
| `interface.py` | Default config: `TycheConfig(tile_size=16, num_rounds=4)`. |
| `__init__.py` | No change. |

---

## Open Questions for Implementation

1. **Key size growth.** T=16, R=4 gives a 4 KB key. This is larger than
   v1 but still fits comfortably in L1. Worth benchmarking key derivation
   (split) cost at this size — the W_r² matmul is now 16×16 instead of
   4×4, which is 64× more FLOPs.

2. **Within-tile independence.** Elements at positions (i,j) and (i',j')
   in the same tile share FMA ancestry. The ALU stage should fully
   decorrelate them, but this should be validated with PractRand on
   interleaved within-tile streams (same test harness as the v1
   split-independence tests).

3. **Embedding convergence sweep.** The recommended first experiment is
   a 2D sweep over `(embedding_mode, num_rounds)` using the existing
   avalanche test suite. This will establish the minimum R needed per
   mode and replace the informal v1 diffusion claims with measured bounds.
