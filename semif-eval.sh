#!/bin/sh
# semif-eval.sh — прогнать SemIf на этой машине и сравнить с опубликованными числами.
#
#   ./semif-eval.sh                 # 144 авторских пункта, backend llama.cpp (CPU, GGUF)
#   SEMIF_BACKEND=torch ./semif-eval.sh   # если освободить VRAM под 4B bf16 (~8.5 GB)
#
# Что делает:
#   1) считает решения на benchmarks/data/authored144.jsonl локальной моделью;
#   2) сравнивает точность с закоммиченными предсказаниями авторов (BF16, Torch, RunPod)
#      на тех же строках — то есть меряет ровно наш дельта-фактор: квант + бэкенд.
set -e

SEMIF="${SEMIF_DIR:-$HOME/Projects/semif}"
GGUF="$SEMIF/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf"
REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GOLD="$SEMIF/benchmarks/data/authored144.jsonl"
REF="$SEMIF/results/raw/predictions/direct-authored144.jsonl"
BACKEND="${SEMIF_BACKEND:-llamacpp}"
OUT="$SEMIF/results/local-authored144-${BACKEND}.jsonl"
THREADS="${SEMIF_THREADS:-$(nproc)}"

[ -x "$SEMIF/.venv/bin/semif-score" ] || { echo "Нет $SEMIF/.venv — см. README SemIf" >&2; exit 1; }
[ -f "$GOLD" ] || { echo "Нет фикстуры: $GOLD" >&2; exit 1; }

if [ "$BACKEND" = "llamacpp" ]; then
    [ -f "$GGUF" ] || { echo "Нет GGUF: $GGUF" >&2; exit 1; }
    SIZE=$(stat -c%s "$GGUF")
    [ "$SIZE" -ge 3000000000 ] || { echo "GGUF недокачан: $SIZE байт из ~3.01 GB" >&2; exit 1; }
fi

rm -f "$OUT"
echo "=== 1/2 замер ($BACKEND, $THREADS потоков) ==="
if [ "$BACKEND" = "llamacpp" ]; then
    "$SEMIF/.venv/bin/semif-score" --mode direct --backend llamacpp \
        --gguf "$GGUF" --llama-threads "$THREADS" \
        --model Qwen/Qwen3.5-4B --revision "$REV" \
        --input "$GOLD" --output "$OUT"
else
    "$SEMIF/.venv/bin/semif-score" --mode direct --backend torch --device cuda \
        --model Qwen/Qwen3.5-4B --revision "$REV" \
        --input "$GOLD" --output "$OUT"
fi

echo
echo "=== 2/2 сравнение ==="
python3 "$(dirname "$0")/semif-compare.py" "$GOLD" "$OUT" "$REF"

echo
echo "Опубликованные числа авторов для ориентира (docs/RESULTS.md):"
echo "  authored144, 4B BF16 direct  : 0.813 balanced accuracy"
echo "  TypeSafe subset, agreement   : 0.845 против 0.883 у облачного Jev"
echo "  MiniCPM5-2B Q4 (браузер)     : 0.686 / 0.637"
