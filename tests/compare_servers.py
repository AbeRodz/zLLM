"""
Comparativa entre servidores de inferencia — Apple M3 Pro
Servidor propio (zllm) vs Ollama, mismo modelo (gemma3 fp16), mismas métricas.
"""

import time
import json
import statistics
import httpx

SERVERS = {
    "zllm": {
        "base_url": "http://localhost:8080/v1",
        "model":    "gemma3",
    },
    "Ollama": {
        "base_url": "http://localhost:11434/v1",
        "model":    "gemma3:latest",
    },
}

N_RUNS    = 5
MAX_TOKS  = 200
TEMP      = 0.0

# Fixed prompts — same across servers
PROMPTS = {
    "corto": "¿Cuál es la capital de Francia?",
    "medio": (
        "Explica el proceso de fotosíntesis incluyendo las reacciones "
        "dependientes e independientes de la luz, el rol de la clorofila "
        "y la ecuación general de la reacción."
    ),
    "largo": (
        "Proporciona un análisis exhaustivo de la Revolución Industrial: "
        "transformación económica, cambios sociales, innovación tecnológica, "
        "impacto ambiental y redes de comercio global. "
        "Incluye diferencias regionales entre Gran Bretaña, Europa continental "
        "y América del Norte, y cómo cada región adaptó las nuevas tecnologías "
        "a sus estructuras económicas y sociales preexistentes."
    ),
}


def measure_stream(base_url: str, model: str, prompt: str) -> dict:
    payload = {
        "model":      model,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKS,
        "temperature": TEMP,
        "stream":     True,
    }
    t_start = time.perf_counter()
    t_first = None
    tok_count = 0

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{base_url}/chat/completions",
                           json=payload,
                           headers={"Content-Type": "application/json"}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if t_first is None:
                            t_first = time.perf_counter()
                        tok_count += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    t_end = time.perf_counter()
    return {
        "ttft_ms": (t_first - t_start) * 1000 if t_first else None,
        "e2e_ms":  (t_end  - t_start) * 1000,
        "tps":     tok_count / (t_end - t_start) if tok_count else 0.0,
        "tokens":  tok_count,
    }


def bench_server(name: str, base_url: str, model: str) -> dict:
    server_results = {}
    print(f"\n{'─'*60}")
    print(f"Servidor: {name}  ({base_url}, modelo={model})")
    print(f"{'─'*60}")

    for label, prompt in PROMPTS.items():
        print(f"\n  Prompt '{label}' — {N_RUNS} ejecuciones:")
        runs = []
        for i in range(N_RUNS):
            r = measure_stream(base_url, model, prompt)
            runs.append(r)
            ttft_str = f"{r['ttft_ms']:.0f}ms" if r["ttft_ms"] else "—"
            print(f"    run {i+1}: TTFT={ttft_str}  TPS={r['tps']:.1f}  E2E={r['e2e_ms']:.0f}ms")

        valid_ttft = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
        server_results[label] = {
            "ttft_mean": statistics.mean(valid_ttft),
            "ttft_std":  statistics.stdev(valid_ttft) if len(valid_ttft) > 1 else 0,
            "tps_mean":  statistics.mean(r["tps"] for r in runs),
            "tps_std":   statistics.stdev(r["tps"] for r in runs) if len(runs) > 1 else 0,
            "e2e_mean":  statistics.mean(r["e2e_ms"] for r in runs),
        }
    return server_results


def print_comparison(all_results: dict):
    """Side-by-side summary table."""
    server_names = list(all_results.keys())
    prompt_labels = list(PROMPTS.keys())

    print("\n\n" + "═" * 78)
    print("COMPARATIVA — Apple M3 Pro, gemma3 fp16, streaming, max_tokens=200")
    print("═" * 78)

    for metric, unit, key_mean, key_std in [
        ("TTFT", "ms",      "ttft_mean", "ttft_std"),
        ("TPS",  "tok/s",   "tps_mean",  "tps_std"),
        ("E2E",  "ms",      "e2e_mean",  None),
    ]:
        print(f"\n  {metric} ({unit}):")
        header = f"    {'Prompt':<10}" + "".join(f"{n:>16}" for n in server_names)
        print(header)
        print("    " + "─" * (10 + 16 * len(server_names)))
        for label in prompt_labels:
            row = f"    {label:<10}"
            for sname in server_names:
                r = all_results[sname][label]
                val = r[key_mean]
                std = r[key_std] if key_std else None
                if std is not None:
                    row += f"{val:>9.1f}±{std:<5.1f}"
                else:
                    row += f"{val:>14.0f}  "
            print(row)

    # Speedup
    if "zllm" in all_results and "Ollama" in all_results:
        print(f"\n  TPS speedup zllm vs Ollama:")
        for label in prompt_labels:
            z = all_results["zllm"][label]["tps_mean"]
            o = all_results["Ollama"][label]["tps_mean"]
            ratio = z / o if o else float("inf")
            print(f"    {label:<10}: {z:.1f} / {o:.1f} = {ratio:.2f}x")


def print_latex(all_results: dict):
    server_names = list(all_results.keys())
    col_spec = "l" + " r r r" * len(server_names)

    print("\n\n% ── LaTeX: Comparativa entre servidores ─────────────────────────────")
    print(r"""\begin{table}[ht]
\centering
\caption{Comparativa de rendimiento entre servidores de inferencia --- Apple M3 Pro, modelo gemma3 (fp16), \textit{streaming}.}
\label{tab:server_comparison}""")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")

    # Header row
    header = "\\textbf{Prompt}"
    for sname in server_names:
        header += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{sname}}}}}"
    print(header + r" \\")

    sub = ""
    for sname in server_names:
        sub += r" & \textbf{TTFT (ms)} & \textbf{TPS (tok/s)} & \textbf{E2E (ms)}"
    print(sub + r" \\")
    print(r"\midrule")

    labels_display = {"corto": "Corto", "medio": "Medio", "largo": "Largo"}
    for key, display in labels_display.items():
        row = display
        for sname in server_names:
            r = all_results[sname][key]
            row += (f" & ${r['ttft_mean']:.0f}\\pm{r['ttft_std']:.0f}$"
                    f" & ${r['tps_mean']:.1f}\\pm{r['tps_std']:.1f}$"
                    f" & ${r['e2e_mean']:.0f}$")
        print(row + r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    print(f"Plataforma: Apple M3 Pro  |  Ejecuciones: {N_RUNS}  |  max_tokens: {MAX_TOKS}")

    all_results = {}
    for sname, cfg in SERVERS.items():
        all_results[sname] = bench_server(sname, cfg["base_url"], cfg["model"])

    print_comparison(all_results)
    print_latex(all_results)
