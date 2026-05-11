"""
Benchmark: decodificación estándar (greedy) vs lookahead, fp16 vs q8_0.
Cubre los 4 modos: fp16×greedy, fp16×lookahead, q8×greedy, q8×lookahead.
Extrae TPS, tokens generados y tasa de aceptación del stderr del CLI.
"""

import subprocess
import re
import statistics

BINARY = "/Users/rodz/Documents/projects/zLLM/zig-out/bin/zLLM"
N_RUNS = 5

PROMPT = (
    "Create a complete implementation of a priority-queue based task scheduler "
    "in Rust. Include struct definitions, methods for push, pop, and peek, "
    "and a main function demonstrating its usage."
)

# (command, extra_args, quant_label, algo_label)
MODES = [
    ("run",            ["gemma3"],        "fp16", "Greedy"),
    ("run-lookahead",  ["gemma3"],        "fp16", "Lookahead"),
    ("run",            ["q8", "gemma3"],  "q8_0", "Greedy"),
    ("run-lookahead",  ["q8", "gemma3"],  "q8_0", "Lookahead"),
]


def run_once(command: str, extra_args: list[str]) -> dict | None:
    cmd = [BINARY, command] + extra_args + [PROMPT]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stderr + result.stdout

    m_speed   = re.search(r"speed:\s*([\d.]+)\s*t/s", output)
    m_decoded = re.search(r"decoded\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s*s", output)
    m_accept  = re.search(r"n_accept\s*=\s*\d+\s*\(([\d.]+)%\s*acceptance", output)

    if not m_speed or not m_decoded:
        print(f"    [WARN] parse failed:\n{output[:300]}")
        return None

    return {
        "tps":         float(m_speed.group(1)),
        "tokens":      int(m_decoded.group(1)),
        "elapsed_s":   float(m_decoded.group(2)),
        "accept_rate": float(m_accept.group(1)) if m_accept else None,
    }


def bench_mode(command: str, extra_args: list[str], label: str) -> list[dict]:
    print(f"\n  [{label}]  {BINARY} {command} {' '.join(extra_args)} ...")
    runs = []
    for i in range(N_RUNS):
        print(f"    run {i+1}/{N_RUNS}...", end=" ", flush=True)
        r = run_once(command, extra_args)
        if r:
            acc = f"  accept={r['accept_rate']:.1f}%" if r["accept_rate"] else ""
            print(f"TPS={r['tps']:.2f}  tokens={r['tokens']}  t={r['elapsed_s']:.2f}s{acc}")
            runs.append(r)
        else:
            print("FAILED")
    return runs


def summarize(runs: list[dict]) -> dict:
    tps  = [r["tps"] for r in runs]
    toks = [r["tokens"] for r in runs]
    acc  = [r["accept_rate"] for r in runs if r["accept_rate"] is not None]
    return {
        "tps_mean":    statistics.mean(tps),
        "tps_std":     statistics.stdev(tps) if len(tps) > 1 else 0.0,
        "tok_mean":    statistics.mean(toks),
        "accept_mean": statistics.mean(acc) if acc else None,
    }


if __name__ == "__main__":
    print(f"Benchmark: {N_RUNS} runs por modo")
    print(f"Prompt: '{PROMPT[:70]}…'\n")
    print("=" * 65)

    all_results: dict[tuple, dict] = {}
    for command, extra_args, quant, algo in MODES:
        label = f"{quant} / {algo}"
        runs = bench_mode(command, extra_args, label)
        all_results[(quant, algo)] = summarize(runs)

    # ── Resumen en consola ────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("RESUMEN — Apple M3 Pro, gemma3")
    print("=" * 65)
    print(f"  {'Cuantización':<8} {'Algoritmo':<12} {'TPS (tok/s)':>16}  "
          f"{'Tokens':>8}  {'Aceptación':>12}")
    print("  " + "─" * 60)
    for (quant, algo), s in all_results.items():
        acc_str = f"{s['accept_mean']:.1f}%" if s["accept_mean"] else "—"
        print(f"  {quant:<8} {algo:<12} "
              f"{s['tps_mean']:>8.2f}±{s['tps_std']:>5.2f}  "
              f"{s['tok_mean']:>8.0f}  {acc_str:>12}")

    # Speedup matrix
    print("\n  Speedups (TPS ratio):")
    base_fp16_g = all_results[("fp16", "Greedy")]["tps_mean"]
    for (quant, algo), s in all_results.items():
        ratio = s["tps_mean"] / base_fp16_g
        print(f"    {quant}/{algo:<12}: {ratio:.2f}× vs fp16/Greedy")
    print("=" * 65)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print("""
% ── LaTeX: Tabla 2×2 cuantización × algoritmo ────────────────────────
\\begin{table}[ht]
\\centering
\\caption{Comparativa de modos de decodificación y cuantización --- Apple M3 Pro, modelo gemma3.}
\\label{tab:quant_algo}
\\begin{tabular}{l l r r r}
\\toprule
\\textbf{Cuantización} & \\textbf{Algoritmo} & \\textbf{TPS (tok/s)} & \\textbf{Tokens generados} & \\textbf{Tasa de aceptación} \\\\
\\midrule""")

    for (quant, algo), s in all_results.items():
        acc_tex = f"${s['accept_mean']:.1f}\\%$" if s["accept_mean"] else "---"
        print(f"{quant} & {algo} & "
              f"${s['tps_mean']:.2f} \\pm {s['tps_std']:.2f}$ & "
              f"${s['tok_mean']:.0f}$ & {acc_tex} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")
