"""
Inference server memory footprint.
One idle measurement + one generation request (max_tokens=300) with RSS polling.
"""
import subprocess
import threading
import time
import httpx

BASE_URL = "http://localhost:8080/v1"
MODEL = "gemma3:f16"
POLL_INTERVAL = 0.2

PROMPT = (
    "Write a detailed technical explanation of how transformer attention "
    "mechanisms work, covering queries, keys, values, and multi-head attention."
)


def get_server_pid() -> int | None:
    for pattern in ["zLLM serve", "zLLM"]:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        pids = [p.strip() for p in r.stdout.strip().split() if p.strip()]
        if pids:
            return int(pids[0])
    r = subprocess.run(["lsof", "-t", "-i", ":8080"], capture_output=True, text=True)
    pids = [p.strip() for p in r.stdout.strip().split() if p.strip()]
    return int(pids[0]) if pids else None


def get_rss_mb(pid: int) -> float:
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    try:
        return int(r.stdout.strip()) / 1024
    except ValueError:
        return 0.0


def stream_request() -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 300,
        "temperature": 0.0,
        "stream": True,
    }
    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{BASE_URL}/chat/completions", json=payload) as r:
            for _ in r.iter_lines():
                pass


if __name__ == "__main__":
    pid = get_server_pid()
    if not pid:
        print("ERROR: no server found on :8080. Start with ./zLLM serve")
        raise SystemExit(1)
    print(f"Server PID: {pid}")

    # Idle RSS
    print("\nSampling idle RSS...")
    idle_samples = [get_rss_mb(pid) for _ in range(8)]
    time.sleep(0.1)
    idle_mean = sum(idle_samples) / len(idle_samples)
    print(f"  Reposo: {idle_mean:.0f} MB  ({idle_mean/1024:.2f} GB)")

    # Active RSS
    print("\nGenerating (max_tokens=300), polling RSS...")
    active_samples: list[float] = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            active_samples.append(get_rss_mb(pid))
            time.sleep(POLL_INTERVAL)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    stream_request()
    stop.set()
    poller.join()

    peak_rss = max(active_samples) if active_samples else idle_mean
    active_mean = sum(active_samples) / len(active_samples) if active_samples else idle_mean

    print(f"  Media activa: {active_mean:.0f} MB  ({active_mean/1024:.2f} GB)")
    print(f"  Pico:         {peak_rss:.0f} MB  ({peak_rss/1024:.2f} GB)")
    print(f"  Overhead:    +{peak_rss - idle_mean:.0f} MB")

    print("\n" + "=" * 55)
    print("RESUMEN — Huella de memoria")
    print("=" * 55)
    print(f"  Reposo (fp16 cargado): {idle_mean:>7.0f} MB  ({idle_mean/1024:.2f} GB)")
    print(f"  Pico durante generación:{peak_rss:>6.0f} MB  ({peak_rss/1024:.2f} GB)")
    print(f"  Overhead activo:       +{peak_rss - idle_mean:>6.0f} MB")
    print("=" * 55)
