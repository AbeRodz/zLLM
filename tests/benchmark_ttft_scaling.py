"""
TTFT vs input context length.
Uses max_tokens=1 so each request is pure prefill — very fast, minimal load.
The three short/medium/long points from the main benchmark are included manually.
"""
import httpx
import time
import statistics
import json

BASE_URL = "http://localhost:8080/v1"
MODEL = "gemma3:f16"
N_RUNS = 3

SENTENCE = (
    "The transformer architecture relies on self-attention mechanisms "
    "to process input sequences in parallel rather than sequentially, "
    "which enables efficient training on modern hardware accelerators. "
)

# Character targets chosen to land at ~250, 600, 1200, 2500 tokens
TARGET_CHARS = [1000, 2500, 5000, 10000]

# Points already measured in the main benchmark (corto/medio/largo)
EXISTING = [
    (6,   117, 15),   # corto:  ~6 tokens,  TTFT=117±15 ms
    (35,  125,  5),   # medio: ~35 tokens,  TTFT=125±5 ms
    (141, 139,  7),   # largo: ~141 tokens, TTFT=139±7 ms
]


def build_prompt(target_chars: int) -> str:
    reps = max(1, target_chars // len(SENTENCE) + 1)
    return (SENTENCE * reps)[:target_chars]


def measure_ttft(prompt: str) -> tuple[float, int]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    first_content = True
    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{BASE_URL}/chat/completions", json=payload) as r:
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if first_content and delta.get("content"):
                    ttft = (time.perf_counter() - t0) * 1000
                    first_content = False
                usage = chunk.get("usage") or {}
                if usage.get("prompt_tokens"):
                    prompt_tokens = usage["prompt_tokens"]
    if ttft is None:
        ttft = (time.perf_counter() - t0) * 1000
    return ttft, prompt_tokens


if __name__ == "__main__":
    print(f"TTFT scaling — {MODEL}, max_tokens=1, {N_RUNS} runs/point")
    print("(corto/medio/largo reused from main benchmark)")
    print("=" * 60)

    new_results = []
    for target in TARGET_CHARS:
        prompt = build_prompt(target)
        ttfts = []
        actual_tokens = 0
        print(f"\n  target ~{target} chars:")
        for i in range(N_RUNS):
            ttft, pt = measure_ttft(prompt)
            ttfts.append(ttft)
            actual_tokens = pt
            print(f"    run {i+1}: TTFT={ttft:.0f}ms  tokens={pt}")
        mean = statistics.mean(ttfts)
        std = statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0
        new_results.append((actual_tokens, mean, std))
        print(f"  => {actual_tokens} tokens: {mean:.0f}±{std:.0f} ms")

    # Merge existing + new, sorted by token count
    all_results = [(pt, m, s) for pt, m, s in EXISTING] + new_results
    all_results.sort(key=lambda x: x[0])

    print("\n\n" + "=" * 60)
    print("RESUMEN — TTFT vs tokens de entrada")
    print("=" * 60)
    print(f"  {'Tokens':>8}  {'TTFT (ms)':>18}")
    print("  " + "-" * 30)
    for pt, mean, std in all_results:
        print(f"  {pt:>8}  {mean:>9.0f} ± {std:>5.0f}")
    print("=" * 60)

    print("""
% ── LaTeX: TTFT scaling ──────────────────────────────────────────────
\\begin{table}[ht]
\\centering
\\caption{TTFT en función de la longitud del contexto de entrada --- Apple M3 Pro, gemma3 (fp16).}
\\label{tab:ttft_scaling}
\\begin{tabular}{r r}
\\toprule
\\textbf{Tokens de entrada} & \\textbf{TTFT (ms)} \\\\
\\midrule""")
    for pt, mean, std in all_results:
        print(f"${pt}$ & ${mean:.0f} \\pm {std:.0f}$ \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
