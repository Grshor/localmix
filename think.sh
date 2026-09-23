#!/bin/sh
# think.sh — локальный ризонер VibeThinker-3B как инструмент командной строки.
#
#   ./think.sh "Сколько целых от 1 до 600 делится на 7, но не на 3?"
#   echo "задача" | ./think.sh                 # или со stdin
#   THINK_RAW=1 ./think.sh "..."               # весь ответ, а не только ANSWER
#   THINK_MAX=4096 ./think.sh "..."            # бюджет новых токенов (по умолчанию 8192)
#
# Почему НЕ через агента OMP: VibeThinker-3B не обучен tool-calling. Как только ему
# показывают схемы инструментов, он начинает галлюцинировать вызовы
# (<function name="edit"> …), тратит на это минуты и засоряет вывод. Замерено:
# через субагента — 4 мин 34 с и мусор в ответе; этим скриптом — 20 с и чистая
# строка ANSWER. Здесь инструментов нет вообще: только системный промпт и задача.
set -e
PORT="${THINK_PORT:-8092}"
MAX="${THINK_MAX:-8192}"

if [ -n "$1" ]; then PROMPT="$1"; else PROMPT="$(cat)"; fi
[ -n "$PROMPT" ] || { echo "Пустая задача." >&2; exit 2; }

SYS='You are a reasoning module trained for problems with checkable answers: mathematics, algorithms, self-contained code, formal logic. Solve the problem below using only the information given. Do not ask questions: there is no follow-up turn. End with a final line of exactly the form "ANSWER: <value>". If the problem needs facts or actions you were not given, end with "UNSUPPORTED: <what is missing>" instead of guessing.'

BODY=$(python3 -c '
import json, sys
sys_prompt, user, mx = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({
    "messages": [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": user}],
    "max_tokens": mx, "temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0,
}))' "$SYS" "$PROMPT" "$MAX")

RESP=$(curl -sf -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' -d "$BODY") || {
    echo "Сервер ризонера не отвечает на 127.0.0.1:$PORT — запусти ./start.sh think" >&2
    exit 3
}

if [ -n "$THINK_RAW" ]; then
    printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
else
    printf '%s' "$RESP" | python3 -c '
import json, sys
content = json.load(sys.stdin)["choices"][0]["message"]["content"]
lines = [l.strip() for l in content.splitlines() if l.strip()]
final = [l for l in lines if l.startswith("ANSWER:") or l.startswith("UNSUPPORTED:")]
print(final[-1] if final else (lines[-1] if lines else ""))'
fi
