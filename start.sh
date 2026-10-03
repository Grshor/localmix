#!/bin/sh
# localmix — три локальных сервера llama.cpp для «смеси моделей» на этой машине.
#
#   fast    : MiniCPM5-2B Q4_K_M        — воркер: 131k контекст, tool-calling, Apache-2.0
#   think   : VibeThinker-3B Q4_K_M     — узкий ризонер: математика/код, MIT, БЕЗ tool-calling
#   decider : decider-4b v2.1 Q4_K_M (Mapika) — Jev-класс: состояние + критерий + варианты →
#                                          буква и вероятность за один проход, Apache-2.0
#
# Почему так:
#   • Порт 8091/8092, а не 8080: 8080 — дефолт Bonsai-demo и порт автообнаружения
#     llama.cpp в OMP. Эти два сервера объявлены в models.yml явно, чтобы не
#     спорить за 8080 с Bonsai.
#   • --min-p 0 обязателен для MiniCPM5-2B: дефолт llama.cpp (0.05) отрезает
#     токены, которыми модель выходит из цикла повторений. Вендор подтвердил
#     (HF discussion #9). Температура 1.0 / top-p 0.95 / top-k 0 — их рецепт.
#   • VibeThinker — длинная цепочка рассуждений: ему нужен большой -c и
#     max_tokens на стороне клиента (модельная карта: до 40 960 новых токенов).
#   • Оба веса целиком в VRAM (-ngl 99), flash attention on, KV-кеш в 8 бит.
#
# Память (замерено, RTX 4070 SUPER 12 282 MiB):
#   fast  -c 32768: веса 1.56 GB + KV ~0.67 GB
#   think -c 40960: веса 1.93 GB + KV ~0.72 GB
#   decider -c 8192: 3.07 GB в VRAM (веса 2.71 GB)
#   все трое + рабочий стол: 10.0–10.1 GB занято, 2.2 GB свободно
#   (замерено 29.09.2026 по процессам: 2360 + 2932 + 3074 MiB).
#   Трое вместе — только если свободно ≥ 4 GB; по умолчанию поднимается пара,
#   решатель — отдельно (./start.sh decider) или всё сразу (./start.sh full).
#   Поднять контекст: FAST_CTX=65536 ./start.sh fast  (следи за nvidia-smi).
#
# Использование:
#   ./start.sh            — рабочая пара: воркер + ризонер
#   ./start.sh fast       — только MiniCPM5-2B   (http://127.0.0.1:8091)
#   ./start.sh think      — только VibeThinker-3B (http://127.0.0.1:8092)
#   ./start.sh decider    — только decider-4b v2.1 (решатель) (http://127.0.0.1:8095)
#   ./start.sh full       — все три
#   ./stop.sh             — остановить
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
LOG_DIR="$SCRIPT_DIR/logs"
RUN_DIR="$SCRIPT_DIR/run"

# Бинарь llama.cpp, собранный под CUDA (тот же, что использует Bonsai-demo).
BIN="${LLAMA_SERVER:-$HOME/Bonsai-demo/bin/cuda/llama-server}"
BIN_DIR="$(cd "$(dirname "$BIN")" && pwd)"
export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

FAST_GGUF="${FAST_GGUF:-$MODELS_DIR/MiniCPM5-2B-Q4_K_M.gguf}"
THINK_GGUF="${THINK_GGUF:-$MODELS_DIR/VibeThinker-3B.Q4_K_M.gguf}"
# Решатель — decider-4b v2.1 (Mapika): не генерирует текст, отдаёт распределение по
# объявленным вариантам. Рядом должен лежать его decider_config.json (температуры).
DECIDER_GGUF="${DECIDER_GGUF:-$MODELS_DIR/decider-4b-v2.1-Q4_K_M.gguf}"
FAST_PORT="${FAST_PORT:-8091}"
THINK_PORT="${THINK_PORT:-8092}"
DECIDER_PORT="${DECIDER_PORT:-8095}"
FAST_CTX="${FAST_CTX:-32768}"
THINK_CTX="${THINK_CTX:-40960}"
DECIDER_CTX="${DECIDER_CTX:-8192}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

[ -x "$BIN" ] || { echo "Нет llama-server: $BIN" >&2; exit 1; }

wait_healthy() {
    _port="$1"
    _i=0
    while [ "$_i" -lt 180 ]; do
        if curl -sf --max-time 2 "http://127.0.0.1:$_port/health" >/dev/null 2>&1; then
            echo "  ✓ http://127.0.0.1:$_port готов"
            return 0
        fi
        _i=$((_i + 1))
        sleep 1
    done
    echo "  ✗ порт $_port не поднялся за 180 с — смотри $LOG_DIR/*.log" >&2
    return 1
}

start_fast() {
    [ -f "$FAST_GGUF" ] || { echo "Нет модели: $FAST_GGUF" >&2; exit 1; }
    if curl -sf --max-time 2 "http://127.0.0.1:$FAST_PORT/health" >/dev/null 2>&1; then
        echo "fast уже запущен на $FAST_PORT"; return 0
    fi
    echo "=== MiniCPM5-2B Q4_K_M (воркер) → :$FAST_PORT ==="
    # --chat-template-kwargs enable_thinking=false: без этого модель уходит в
    # <think> и тратит весь бюджет на рассуждение, отдавая ПУСТОЙ content —
    # тихая поломка на стороне клиента. С false шаблон вставляет пустой блок
    # <think></think> и модель отвечает сразу. Запрос всё ещё может включить
    # мышление: request-level chat_template_kwargs перекрывает серверные.
    "$BIN" -m "$FAST_GGUF" --host 127.0.0.1 --port "$FAST_PORT" \
        -ngl 99 -fa on -c "$FAST_CTX" -np 1 --jinja -ctk q8_0 -ctv q8_0 \
        --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0 \
        --chat-template-kwargs '{"enable_thinking": false}' \
        >"$LOG_DIR/fast.log" 2>&1 &
    echo $! >"$RUN_DIR/fast.pid"
    wait_healthy "$FAST_PORT"
}

start_think() {
    [ -f "$THINK_GGUF" ] || { echo "Нет модели: $THINK_GGUF" >&2; exit 1; }
    if curl -sf --max-time 2 "http://127.0.0.1:$THINK_PORT/health" >/dev/null 2>&1; then
        echo "think уже запущен на $THINK_PORT"; return 0
    fi
    echo "=== VibeThinker-3B Q4_K_M (ризонер) → :$THINK_PORT ==="
    "$BIN" -m "$THINK_GGUF" --host 127.0.0.1 --port "$THINK_PORT" \
        -ngl 99 -fa on -c "$THINK_CTX" -np 1 --jinja -ctk q8_0 -ctv q8_0 \
        --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0 \
        >"$LOG_DIR/think.log" 2>&1 &
    echo $! >"$RUN_DIR/think.pid"
    wait_healthy "$THINK_PORT"
}

start_decider() {
    [ -f "$DECIDER_GGUF" ] || {
        echo "Нет модели: $DECIDER_GGUF" >&2
        echo "Скачать: huggingface.co/Mapika/decider-4b-GGUF → models/ (файл и decider_config.json)" >&2
        exit 1
    }
    if curl -sf --max-time 2 "http://127.0.0.1:$DECIDER_PORT/health" >/dev/null 2>&1; then
        echo "decider уже запущен на $DECIDER_PORT"; return 0
    fi
    echo "=== decider-4b v2.1 Q4_K_M (решатель) → :$DECIDER_PORT ==="
    # Без --jinja: у чекпойнта в конфиге "layout": "plain", он спрашивается сырым
    # промптом через /completion (см. decide.py), а не через chat-шаблон. Шаблон
    # Qwen3.5 добавил бы <|im_start|> и сломал бы раскладку: первым токеном стала бы
    # не буква варианта, а вероятность читается именно из логитов этой буквы.
    # --temp 0: выбор буквы должен быть воспроизводимым; логиты от температуры не
    # зависят (проверено: при temperature 0 и 1.0 в запросе они совпадают).
    "$BIN" -m "$DECIDER_GGUF" --host 127.0.0.1 --port "$DECIDER_PORT" \
        -ngl 99 -fa on -c "$DECIDER_CTX" -np 1 -ctk q8_0 -ctv q8_0 \
        --temp 0 \
        >"$LOG_DIR/decider.log" 2>&1 &
    echo $! >"$RUN_DIR/decider.pid"
    wait_healthy "$DECIDER_PORT"
}

case "${1:-all}" in
    fast)    start_fast ;;
    think)   start_think ;;
    decider) start_decider ;;
    all)     start_fast; start_think ;;
    full)    start_fast; start_think; start_decider ;;
    *)       echo "Использование: $0 [fast|think|decider|all|full]" >&2; exit 2 ;;
esac
