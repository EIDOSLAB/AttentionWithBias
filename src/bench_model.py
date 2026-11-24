import time
import contextlib
import torch

from torch.utils.flop_counter import FlopCounterMode

from torch.nn.attention import SDPBackend, sdpa_kernel
import statistics



def _amp_ctx(device, amp_dtype):
    """autocast only if requested and on cuda"""
    if amp_dtype is not None and device.type == "cuda":
        return torch.autocast("cuda", dtype=amp_dtype)
    return contextlib.nullcontext()



@torch.no_grad()
def benchmark_model(model, x, warmup=50, iters=200, amp_dtype=None,
                    backend=None):

    model.eval()
    device = x.device

    if backend is not None:
        backend_ctx = lambda: sdpa_kernel([backend])
    else:
        backend_ctx = contextlib.nullcontext

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with _amp_ctx(device, amp_dtype):
        with backend_ctx():
            for _ in range(warmup):
                y = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

            with backend_ctx():
                for i in range(iters):
                    starts[i].record()
                    y = model(x)
                    ends[i].record()

            torch.cuda.synchronize()

            times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            avg_ms = statistics.mean(times_ms)
            std_ms = statistics.stdev(times_ms) if iters > 1 else 0.0

        else:
            times_ms = []
            with backend_ctx():
                for _ in range(iters):
                    t0 = time.perf_counter()
                    y = model(x)
                    times_ms.append((time.perf_counter() - t0) * 1000)
            avg_ms = statistics.mean(times_ms)
            std_ms = statistics.stdev(times_ms) if iters > 1 else 0.0

    return avg_ms, std_ms, y

@torch.no_grad()
def compute_flops(model, x, warmup=10, amp_dtype=None):
    model.eval()
    device = x.device

    with _amp_ctx(device, amp_dtype):
        for _ in range(warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    with _amp_ctx(device, amp_dtype):
        with FlopCounterMode(display=False) as fc:
            model(x)

    return fc.get_total_flops()


@torch.no_grad()
def detect_sdpa_backend(model, x, amp_dtype=None):
    model.eval()
    device = x.device

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # warmup
    with _amp_ctx(device, amp_dtype):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    with _amp_ctx(device, amp_dtype):
        with torch.profiler.profile(activities=activities) as prof:
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    names = [e.name.lower() for e in prof.events()]
    found = set()
    for n in names:
        if "scaled_dot_product" not in n and "attention" not in n:
            continue
        if "flash" in n:
            found.add("flash")
        elif "efficient" in n:
            found.add("efficient")
        elif "cudnn" in n:
            found.add("cudnn")
        elif "math" in n or "bmm" in n:
            found.add("math")
    return found or {"unknown"}
