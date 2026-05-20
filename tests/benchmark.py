"""
Inference server benchmark — Apple M3 Pro / gemma3 + smollm2-360
Measures:
  - HTTP streaming: TTFT, TPS, E2E latency across input sizes, RPS
  - CLI quantization: fp16 vs q8_0 greedy TPS across prompt sizes
"""

import time
import statistics
import concurrent.futures
import subprocess
import re
import httpx
import json

BASE_URL = "http://localhost:8080/v1"
HTTP_MODELS_GEMMA3 = {
    "gemma3_fp16": "gemma3:f16",
    "gemma3_q8_0": "gemma3:q8",
}
HTTP_MODELS_SMOLLM2 = {
    "smollm2_fp16": "smollm2-360",
    "smollm2_q8_0": "smollm2-360:q8",
}
HTTP_MODELS = HTTP_MODELS_SMOLLM2  # switch to HTTP_MODELS_GEMMA3 to re-run gemma3
N_RUNS = 5
MAX_TOKENS = 200
TEMPERATURE = 0.0  # deterministic for reproducibility

# ── Prompts of increasing input length ────────────────────────────────────────

PROMPTS_ES = {
    "corto": "¿Cuál es la capital de Francia?",
    "medio": (
        "Explica detalladamente el proceso de fotosíntesis en las plantas, "
        "incluyendo las reacciones dependientes e independientes de la luz, "
        "el rol de la clorofila y cómo las plantas convierten CO2 y agua en "
        "glucosa y oxígeno."
    ),
    "largo": (
        "Eres un asistente experto. Proporciona un análisis exhaustivo sobre "
        "la Revolución Industrial y su impacto en la sociedad moderna. "
        "Cubre los siguientes aspectos: "
        "1) Transformación económica: cómo los sistemas de fábricas reemplazaron "
        "las industrias artesanales, efectos en los mercados laborales y la acumulación "
        "de capital. "
        "2) Cambios sociales: aceleración de la urbanización, cambios en estructuras "
        "familiares, dinámicas de clase y condiciones de vida para los trabajadores. "
        "3) Innovación tecnológica: inventos clave como la máquina de vapor, la "
        "lanzadera volante y el telar mecánico, y cómo generaron efectos en cascada "
        "en la industria textil y metalúrgica. "
        "4) Impacto ambiental: primeras señales de contaminación industrial y "
        "agotamiento de recursos naturales. "
        "5) Redes de comercio global: cómo la industrialización transformó el comercio "
        "internacional y el colonialismo. "
        "Analiza también las diferencias regionales en la adopción industrial entre "
        "Gran Bretaña, Europa continental y América del Norte."
    ),
}

PROMPTS_EN = {
    "corto": "What is the capital of France?",
    "medio": (
        "Explain in detail the process of photosynthesis in plants, "
        "including the light-dependent and light-independent reactions, "
        "the role of chlorophyll, and how plants convert CO2 and water "
        "into glucose and oxygen."
    ),
    "largo": (
        "You are an expert assistant. Provide a comprehensive analysis of "
        "the Industrial Revolution and its impact on modern society. "
        "Cover the following aspects: "
        "1) Economic transformation: how factory systems replaced cottage industries, "
        "effects on labor markets and capital accumulation. "
        "2) Social changes: accelerated urbanization, shifts in family structures, "
        "class dynamics and living conditions for workers. "
        "3) Technological innovation: key inventions such as the steam engine, the "
        "flying shuttle and the power loom, and how they created cascading effects "
        "in the textile and metallurgical industries. "
        "4) Environmental impact: early signs of industrial pollution and depletion "
        "of natural resources. "
        "5) Global trade networks: how industrialization transformed international "
        "trade and colonialism. "
        "Also analyze regional differences in industrial adoption between "
        "Great Britain, continental Europe and North America."
    ),
}

PROMPTS = PROMPTS_EN  # switch to PROMPTS_ES for gemma3


# ── Core measurement ───────────────────────────────────────────────────────────

def measure_stream(prompt: str, model: str) -> dict:
    """Stream a single request; return TTFT (ms), E2E (ms), TPS, token count."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first = None
    token_count = 0

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{BASE_URL}/chat/completions",
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
                        token_count += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    t_end = time.perf_counter()

    ttft = (t_first - t_start) * 1000 if t_first else None
    e2e  = (t_end  - t_start) * 1000
    tps  = token_count / (t_end - t_start) if token_count else 0.0

    return {"ttft_ms": ttft, "e2e_ms": e2e, "tps": tps, "tokens": token_count}


def measure_non_stream(prompt: str, model: str) -> dict:
    """Non-streaming request; returns only E2E and approximate TPS from usage."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    with httpx.Client(timeout=120.0) as client:
        t_start = time.perf_counter()
        resp = client.post(f"{BASE_URL}/chat/completions",
                           json=payload,
                           headers={"Content-Type": "application/json"})
        t_end = time.perf_counter()
    resp.raise_for_status()
    body = resp.json()
    tokens = len(body["choices"][0]["message"]["content"].split())  # approx
    e2e = (t_end - t_start) * 1000
    tps = tokens / (t_end - t_start)
    return {"e2e_ms": e2e, "tps": tps, "tokens": tokens}


# ── RPS under concurrency ──────────────────────────────────────────────────────

def rps_worker(model: str):
    """Single worker for RPS test — short prompt, low max_tokens."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{BASE_URL}/chat/completions",
                           json=payload,
                           headers={"Content-Type": "application/json"})
    return resp.status_code == 200, time.perf_counter() - t0


def measure_rps(model: str, n_requests: int = 20, max_workers: int = 4) -> dict:
    print(f"\n[RPS] {n_requests} solicitudes con {max_workers} workers concurrentes...")
    t_wall_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(rps_worker, model) for _ in range(n_requests)]
        for f in concurrent.futures.as_completed(futures):
            ok, latency = f.result()
            results.append((ok, latency))
    t_wall_end = time.perf_counter()

    completed = sum(1 for ok, _ in results if ok)
    wall_time = t_wall_end - t_wall_start
    rps = completed / wall_time
    avg_lat = statistics.mean(lat for _, lat in results) * 1000
    print(f"  Completadas: {completed}/{n_requests} | Tiempo: {wall_time:.2f}s | RPS: {rps:.2f} | Latencia media: {avg_lat:.0f}ms")
    return {"rps": rps, "completed": completed, "wall_time_s": wall_time, "avg_latency_ms": avg_lat}


# ── Run all benchmarks ─────────────────────────────────────────────────────────

def run_decoding_benchmark(model: str) -> dict:
    results = {}
    for label, prompt in PROMPTS.items():
        word_count = len(prompt.split())
        print(f"\n[Streaming/{model}] Prompt '{label}' (~{word_count} palabras), {N_RUNS} ejecuciones:")
        runs = []
        for i in range(N_RUNS):
            r = measure_stream(prompt, model)
            runs.append(r)
            print(f"  run {i+1}: TTFT={r['ttft_ms']:.0f}ms  TPS={r['tps']:.1f}  E2E={r['e2e_ms']:.0f}ms  tokens={r['tokens']}")

        valid_ttft = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
        results[label] = {
            "ttft_mean":  statistics.mean(valid_ttft),
            "ttft_std":   statistics.stdev(valid_ttft) if len(valid_ttft) > 1 else 0,
            "tps_mean":   statistics.mean(r["tps"] for r in runs),
            "tps_std":    statistics.stdev(r["tps"] for r in runs) if len(runs) > 1 else 0,
            "e2e_mean":   statistics.mean(r["e2e_ms"] for r in runs),
            "e2e_std":    statistics.stdev(r["e2e_ms"] for r in runs) if len(runs) > 1 else 0,
            "avg_tokens": statistics.mean(r["tokens"] for r in runs),
        }
    return results


def print_summary(results: dict, model: str):
    print("\n" + "═" * 70)
    print(f"RESUMEN — Apple M3 Pro, {model}, streaming, max_tokens={MAX_TOKENS}")
    print("═" * 70)
    print(f"{'Prompt':<10} {'TTFT (ms)':>14} {'TPS (tok/s)':>14} {'E2E (ms)':>12} {'Tokens':>8}")
    print("─" * 70)
    for label, r in results.items():
        print(f"{label:<10} "
              f"{r['ttft_mean']:>8.0f}±{r['ttft_std']:>4.0f} "
              f"{r['tps_mean']:>8.1f}±{r['tps_std']:>4.1f} "
              f"{r['e2e_mean']:>10.0f} "
              f"{r['avg_tokens']:>8.0f}")
    print("═" * 70)


def print_latex(results: dict, rps_result: dict, model: str):
    base  = model.split(":")[0]                          # "gemma3", "smollm2-360"
    quant = "q8\\_0" if ":q8" in model else "fp16"
    label_suffix = f"{base} ({quant})"
    table_label  = f"tab:decoding_perf_{base.replace('-','_')}_{quant.replace('_','').replace(chr(92),'')}"
    print("\n\n% ── LaTeX: Tabla de desempeño de decodificación ──────────────────────")
    print(f"\\begin{{table}}[ht]")
    print(f"\\centering")
    print(f"\\caption{{Desempeño de decodificación --- Apple M3 Pro, modelo {label_suffix}, \\textit{{streaming}}, \\texttt{{max\\_tokens}}={MAX_TOKENS}.}}")
    print(f"\\label{{{table_label}}}")
    print(r"""\begin{tabular}{l r r r r}
\toprule
\textbf{Longitud de prompt} & \textbf{TTFT (ms)} & \textbf{TPS (tok/s)} & \textbf{Latencia E2E (ms)} & \textbf{Tokens generados} \\
\midrule""")
    labels = {"corto": "Corto", "medio": "Medio", "largo": "Largo"}
    for key, display in labels.items():
        r = results[key]
        print(f"{display} & "
              f"${r['ttft_mean']:.0f} \\pm {r['ttft_std']:.0f}$ & "
              f"${r['tps_mean']:.1f} \\pm {r['tps_std']:.1f}$ & "
              f"${r['e2e_mean']:.0f}$ & "
              f"${r['avg_tokens']:.0f}$ \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n% ── LaTeX: RPS ───────────────────────────────────────────────────────")
    print(f"% RPS={rps_result['rps']:.2f}, avg latency={rps_result['avg_latency_ms']:.0f}ms, "
          f"completadas={rps_result['completed']}/20, workers=4")


BINARY = "/Users/rodz/Documents/projects/zLLM/zig-out/bin/zLLM"

CLI_PROMPTS_ES = {
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
        "y América del Norte."
    ),
}

CLI_PROMPTS_EN = {
    "corto": "What is the capital of France?",
    "medio": (
        "Explain the process of photosynthesis including the light-dependent "
        "and light-independent reactions, the role of chlorophyll, "
        "and the general equation of the reaction."
    ),
    "largo": (
        "Provide a comprehensive analysis of the Industrial Revolution: "
        "economic transformation, social changes, technological innovation, "
        "environmental impact and global trade networks. "
        "Include regional differences between Great Britain, continental Europe "
        "and North America."
    ),
}

CLI_PROMPTS = CLI_PROMPTS_EN  # switch to CLI_PROMPTS_ES for gemma3

# (quant_label, args_before_prompt)
CLI_QUANTS_GEMMA3 = [
    ("fp16", ["gemma3"]),
    ("q8_0", ["q8", "gemma3"]),
]
CLI_QUANTS_SMOLLM2 = [
    ("smollm2_fp16", ["smollm2-360"]),
    ("smollm2_q8_0", ["q8", "smollm2-360"]),
]
CLI_QUANTS = CLI_QUANTS_SMOLLM2  # switch to CLI_QUANTS_GEMMA3 to re-run gemma3


def cli_run_once(extra_args: list[str], prompt: str) -> dict | None:
    result = subprocess.run(
        [BINARY, "run"] + extra_args + [prompt],
        capture_output=True, text=True, timeout=300,
    )
    output = result.stderr + result.stdout
    m_speed   = re.search(r"speed:\s*([\d.]+)\s*t/s", output)
    m_decoded = re.search(r"decoded\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s*s", output)
    if not m_speed or not m_decoded:
        return None
    return {
        "tps":       float(m_speed.group(1)),
        "tokens":    int(m_decoded.group(1)),
        "elapsed_s": float(m_decoded.group(2)),
    }


def run_cli_quant_benchmark() -> dict:
    """Compare fp16 vs q8_0 greedy TPS via CLI across prompt sizes."""
    results: dict[str, dict[str, dict]] = {}
    for quant_label, extra_args in CLI_QUANTS:
        results[quant_label] = {}
        for prompt_label, prompt in CLI_PROMPTS.items():
            print(f"\n[CLI {quant_label}] Prompt '{prompt_label}', {N_RUNS} ejecuciones:")
            runs = []
            for i in range(N_RUNS):
                r = cli_run_once(extra_args, prompt)
                if r:
                    print(f"  run {i+1}: TPS={r['tps']:.2f}  tokens={r['tokens']}  t={r['elapsed_s']:.2f}s")
                    runs.append(r)
                else:
                    print(f"  run {i+1}: FAILED")
            if runs:
                results[quant_label][prompt_label] = {
                    "tps_mean": statistics.mean(r["tps"] for r in runs),
                    "tps_std":  statistics.stdev(r["tps"] for r in runs) if len(runs) > 1 else 0.0,
                    "tok_mean": statistics.mean(r["tokens"] for r in runs),
                }
    return results


def print_cli_summary(cli_results: dict, fp16_key: str = "fp16", q8_key: str = "q8_0", title: str = "gemma3"):
    print("\n" + "═" * 68)
    print(f"CLI — {title}, fp16 vs q8_0, decodificación greedy, Apple M3 Pro")
    print("═" * 68)
    print(f"  {'Prompt':<8}  {'fp16 TPS':>12}  {'q8_0 TPS':>12}  {'Speedup':>10}")
    print("  " + "─" * 50)
    for label in CLI_PROMPTS:
        fp = cli_results[fp16_key][label]
        q8 = cli_results[q8_key][label]
        ratio = q8["tps_mean"] / fp["tps_mean"]
        print(f"  {label:<8}  "
              f"{fp['tps_mean']:>6.2f}±{fp['tps_std']:>4.2f}  "
              f"{q8['tps_mean']:>6.2f}±{q8['tps_std']:>4.2f}  "
              f"{ratio:>9.2f}×")
    print("═" * 68)


def print_cli_latex(cli_results: dict, fp16_key: str = "fp16", q8_key: str = "q8_0", table_label: str = "tab:quant_tps", caption_model: str = "gemma3"):
    print(f"\n% ── LaTeX: {caption_model} fp16 vs q8_0 CLI TPS ──────────────────────")
    print(f"\\begin{{table}}[ht]")
    print(f"\\centering")
    print(f"\\caption{{Comparativa de TPS entre cuantizaciones fp16 y q8\\_0 --- Apple M3 Pro, {caption_model}, decodificación greedy.}}")
    print(f"\\label{{{table_label}}}")
    print(r"""\begin{tabular}{l r r r}
\toprule
\textbf{Longitud de prompt} & \textbf{fp16 (tok/s)} & \textbf{q8\_0 (tok/s)} & \textbf{Speedup} \\
\midrule""")
    labels = {"corto": "Corto", "medio": "Medio", "largo": "Largo"}
    for key, display in labels.items():
        fp = cli_results[fp16_key][key]
        q8 = cli_results[q8_key][key]
        ratio = q8["tps_mean"] / fp["tps_mean"]
        print(f"{display} & "
              f"${fp['tps_mean']:.2f} \\pm {fp['tps_std']:.2f}$ & "
              f"${q8['tps_mean']:.2f} \\pm {q8['tps_std']:.2f}$ & "
              f"${ratio:.2f}\\times$ \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    print(f"Servidor: {BASE_URL}  |  Plataforma: Apple M3 Pro")
    print(f"Ejecuciones por prompt: {N_RUNS}  |  max_tokens: {MAX_TOKENS}  |  temperature: {TEMPERATURE}\n")

    # ── HTTP streaming benchmark for each quantization ────────────────────────
    all_http: dict[str, dict] = {}
    all_rps:  dict[str, dict] = {}
    for quant_label, model in HTTP_MODELS.items():
        print(f"\n{'═'*70}")
        print(f"HTTP STREAMING — {model}")
        print(f"{'═'*70}")
        all_http[quant_label] = run_decoding_benchmark(model)
        all_rps[quant_label]  = measure_rps(model, n_requests=20, max_workers=4)
        print_summary(all_http[quant_label], model)
        print_latex(all_http[quant_label], all_rps[quant_label], model)

    # ── CLI quantization comparison (greedy, across prompt sizes) ─────────────
    print("\n\n" + "═" * 68)
    print("CLI — fp16 vs q8_0, decodificación greedy")
    print("═" * 68)
    cli_results = run_cli_quant_benchmark()
    print_cli_summary(cli_results, fp16_key="smollm2_fp16", q8_key="smollm2_q8_0", title="smollm2-360")
    print_cli_latex(cli_results,   fp16_key="smollm2_fp16", q8_key="smollm2_q8_0", table_label="tab:quant_tps_smollm2", caption_model="smollm2-360")
