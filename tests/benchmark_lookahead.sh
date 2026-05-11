#!/usr/bin/env bash
# Benchmark: standard greedy decoding vs lookahead decoding
# Captures the "speed: X t/s" line printed to stderr by each CLI mode.

BINARY="/Users/rodz/Documents/projects/zLLM/zig-out/bin/zLLM"
MODEL="gemma3"
N_RUNS=5

PROMPT="Create a complete implementation of a priority-queue based task scheduler in Rust. Include struct definitions, methods for push, pop, and peek, and a main function demonstrating its usage."

echo "================================================================"
echo "Benchmark: decodificación estándar (greedy) vs lookahead"
echo "Modelo: $MODEL  |  Ejecuciones: $N_RUNS"
echo "================================================================"

echo ""
echo "=== run (greedy) ==="
for i in $(seq 1 $N_RUNS); do
  echo -n "  run $i: "
  "$BINARY" run "$MODEL" "$PROMPT" 2>&1 >/dev/null | grep -E "speed:|decoded"
done

echo ""
echo "=== run-lookahead ==="
for i in $(seq 1 $N_RUNS); do
  echo -n "  run $i: "
  "$BINARY" run-lookahead "$MODEL" "$PROMPT" 2>&1 >/dev/null | grep -E "speed:|decoded|accept|W ="
done

echo ""
echo "================================================================"
echo "Extracción de TPS (tok/s):"
echo ""
echo "Greedy TPS:"
for i in $(seq 1 $N_RUNS); do
  "$BINARY" run "$MODEL" "$PROMPT" 2>&1 >/dev/null | grep "speed:" | grep -oE "[0-9]+\.[0-9]+ t/s"
done

echo ""
echo "Lookahead TPS:"
for i in $(seq 1 $N_RUNS); do
  "$BINARY" run-lookahead "$MODEL" "$PROMPT" 2>&1 >/dev/null | grep "speed:" | grep -oE "[0-9]+\.[0-9]+ t/s"
done
