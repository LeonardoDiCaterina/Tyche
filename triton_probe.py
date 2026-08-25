import torch
import triton
import triton.language as tl
import time

@triton.jit
def round_kernel(
    x_ptr, w_ptr, out_ptr,
    T: tl.constexpr,
    ODD_MULT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_i = tl.arange(0, T)
    offs_j = tl.arange(0, T)

    x_base = x_ptr + pid * T * T
    w_base = w_ptr + pid * T * T
    out_base = out_ptr + pid * T * T

    x = tl.load(x_base + offs_i[:, None] * T + offs_j[None, :])
    w = tl.load(w_base + offs_i[:, None] * T + offs_j[None, :])

    # RED-TEAM FIX 1: We MUST pass native int8 to the dot product. 
    # The author's int32 cast is physically impossible to compile.
    w_i32 = w.to(tl.int32)
    acc = tl.dot(x, x, out_dtype=tl.int32) + w_i32

    acc_u32 = acc.to(tl.uint32, bitcast=True)
    acc_u32 = acc_u32 * ODD_MULT
    mixed = acc_u32 ^ (acc_u32 >> 16)

    result = mixed.to(tl.int8)
    tl.store(out_base + offs_i[:, None] * T + offs_j[None, :], result)


def run_round(x, w, T, odd_mult=0x94D049BB):
    N = x.shape[0]
    out = torch.empty_like(x)
    grid = (N,)
    round_kernel[grid](x, w, out, T=T, ODD_MULT=odd_mult)
    return out


def dump_asm(T=32):
    x = torch.randint(-127, 127, (128, T, T), dtype=torch.int8, device="cuda")
    w = torch.randint(-127, 127, (128, T, T), dtype=torch.int32, device="cuda")
    compiled = round_kernel.warmup(
        x, w, torch.empty_like(x), T=T, ODD_MULT=0x94D049BB, grid=(128,)
    )
    print("=== SASS (Skipping PTX for brevity) ===")
    print(compiled.asm.get("sass", "no sass captured -- try cuobjdump"))


def sweep_batch_sizes(T=32, sizes=(1, 8, 64, 512, 4096, 32768, 262144), iters=50):
    print(f"{'N':>8}  {'ms/round':>10}  {'tiles/sec':>14}  {'GB/s (approx)':>14}")
    for N in sizes:
        x = torch.randint(-127, 127, (N, T, T), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 127, (N, T, T), dtype=torch.int32, device="cuda")

        for _ in range(5):  # warmup
            x = run_round(x, w, T)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            x = run_round(x, w, T)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        ms_per_round = 1000 * elapsed / iters
        tiles_per_sec = N * iters / elapsed
        bytes_per_round = N * T * T * (1 + 4)  # int8 in/out + int32 W read
        gbps = bytes_per_round * iters / elapsed / 1e9

        print(f"{N:>8}  {ms_per_round:>10.4f}  {tiles_per_sec:>14,.0f}  {gbps:>14.2f}")


def philox_baseline(sizes=(1_000, 100_000, 10_000_000, 1_000_000_000), iters=20):
    import torch
    print(f"{'n_elements':>14}  {'ms':>10}  {'GB/s':>10}")
    for n in sizes:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            out = torch.randint(0, 2**31, (n,), dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        ms = 1000 * elapsed / iters
        gbps = (n * 4 * iters) / elapsed / 1e9
        print(f"{n:>14,}  {ms:>10.4f}  {gbps:>10.2f}")

if __name__ == "__main__":
    print(">>> Step A: dumping asm (Forced T=32 to allow compilation)")
    dump_asm(T=32)

    print("\n>>> Step B: batch size sweep (Forced T=32)")
    sweep_batch_sizes(T=32)

    print("\n>>> Step C: Philox baseline on same device")
    philox_baseline()