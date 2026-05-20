"""
Concurrency scaling: RPS and latency at 1, 2, 4, 8 concurrent workers.
4-worker result is injected from the main benchmark to avoid redundancy.
15 requests per new level, max_tokens=100 to keep generation short.
"""
import httpx
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8080/v1"
MODEL = "gemma3:f16"
TOTAL_REQUESTS = 15
MAX_TOKENS = 100

# Workers to actually run (4 is already known from main benchmark)
WORKER_COUNTS = [1, 2, 8]

# Existing data point from benchmark.py (4 workers, 20 requests, max_tokens=200)
# RPS=10.60, lat_mean≈372ms — injected directly, not rerun
EXISTING_4W = {"workers": 4, "rps": 10.60, "lat_mean": 372, "lat_std": 0, "lat_p95": None}

PROMPT = (
    "Explain the concept of attention mechanisms in transformer models "
    "in two concise sentences."
)


def single_request() -> float:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{BASE_URL}/chat/completions", json=payload)
        r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def benchmark_level(n_workers: int) -> dict:
    latencies: list[float] = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(single_request) for _ in range(TOTAL_REQUESTS)]
        for f in as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception as e:
                print(f"    [ERROR] {e}")
    elapsed = time.perf_counter() - t_start
    latencies.sort()
    p95_idx = max(0, int(0.95 * len(latencies)) - 1)
    return {
        "workers":   n_workers,
        "rps":       len(latencies) / elapsed,
        "lat_mean":  statistics.mean(latencies) if latencies else 0.0,
        "lat_std":   statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "lat_p95":   latencies[p95_idx] if latencies else 0.0,
    }


if __name__ == "__main__":
    print(f"Concurrency scaling — {MODEL}, {TOTAL_REQUESTS} req/level, max_tokens={MAX_TOKENS}")
    print("(4-worker point reused from main benchmark)")
    print("=" * 65)

    new_results = []
    for w in WORKER_COUNTS:
        print(f"\n  [{w} worker{'s' if w > 1 else ''}] {TOTAL_REQUESTS} requests...")
        r = benchmark_level(w)
        new_results.append(r)
        print(f"    RPS: {r['rps']:.2f}  Lat media: {r['lat_mean']:.0f}ms  "
              f"p95: {r['lat_p95']:.0f}ms")

    # Merge and sort
    all_results = new_results + [EXISTING_4W]
    all_results.sort(key=lambda x: x["workers"])

    print("\n\n" + "=" * 65)
    print("RESUMEN — Escalado de concurrencia")
    print("=" * 65)
    print(f"  {'Workers':>8}  {'RPS':>8}  {'Lat. media (ms)':>18}  {'Lat. p95 (ms)':>14}")
    print("  " + "-" * 55)
    for r in all_results:
        p95_str = f"{r['lat_p95']:.0f}" if r["lat_p95"] else "—"
        print(f"  {r['workers']:>8}  {r['rps']:>8.2f}  "
              f"{r['lat_mean']:>12.0f} ± {r.get('lat_std', 0):>4.0f}  "
              f"{p95_str:>14}")
    print("=" * 65)

    base = all_results[0]["rps"]
    print("\n  Speedup de RPS vs 1 worker:")
    for r in all_results:
        print(f"    {r['workers']:>2}w: {r['rps']/base:.2f}×")

    print("""
% ── LaTeX: concurrency scaling ───────────────────────────────────────
\\begin{table}[ht]
\\centering
\\caption{Escalado de rendimiento bajo concurrencia --- Apple M3 Pro, gemma3 (fp16).}
\\label{tab:concurrency_scaling}
\\begin{tabular}{r r r r}
\\toprule
\\textbf{Clientes concurrentes} & \\textbf{RPS} & \\textbf{Latencia media (ms)} & \\textbf{Latencia p95 (ms)} \\\\
\\midrule""")
    for r in all_results:
        p95_tex = f"${r['lat_p95']:.0f}$" if r["lat_p95"] else "---"
        print(f"${r['workers']}$ & ${r['rps']:.2f}$ & "
              f"${r['lat_mean']:.0f}$ & {p95_tex} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
