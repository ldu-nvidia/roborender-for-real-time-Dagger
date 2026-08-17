# Runs on the warm image3f server: times PyTorch DiT block-stack at batch sizes 1..16.
import inspect, time, torch

pipe = session.pipe
blocks = pipe.dit.blocks
dev0 = next(blocks.parameters()).device
print("blocks device:", dev0, "n_blocks:", len(blocks))
sig = str(inspect.signature(blocks[0].forward))
print("block.forward signature:", sig)

if dev0.type != "cuda":
    blocks.to("cuda")
    print("moved blocks to cuda")

T, C, CTX = 2340, 1536, 1283
fc = torch.randn(T, 1, 64, device="cuda", dtype=torch.float32)
fs = torch.randn(T, 1, 64, device="cuda", dtype=torch.float32)

def run_stack(B, iters=5, warmup=2):
    x = torch.randn(B, T, C, device="cuda", dtype=torch.bfloat16)
    ctx = torch.randn(B, CTX, C, device="cuda", dtype=torch.bfloat16)
    tm = torch.randn(B, 6, C, device="cuda", dtype=torch.bfloat16)
    def once():
        h = x
        with torch.no_grad():
            for blk in blocks:
                h = blk(h, ctx, tm, (fc, fs)) if "freqs" in sig else blk(h, ctx, tm, fc, fs)
                if isinstance(h, tuple):
                    h = h[0]
        return h
    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

RESULT["signature"] = sig
timings = {}
for B in [1, 2, 4, 8, 16]:
    try:
        ms = run_stack(B)
        timings[B] = round(ms, 2)
        print(f"B={B}: {ms:.1f} ms/stack-pass, {ms/B:.1f} ms/sample")
    except Exception as e:
        timings[B] = f"ERR {type(e).__name__}: {str(e)[:200]}"
        print(f"B={B} failed: {e}")
        break
RESULT["stack_ms"] = timings
if dev0.type != "cuda":
    blocks.to(dev0)
torch.cuda.empty_cache()
