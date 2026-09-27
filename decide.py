#!/usr/bin/env python3
"""decide.py — Jev-класс локально: состояние + критерий + варианты → буква и вероятности.

Формат запроса не выдуман заново, а взят у харнесса SemIf (`~/Projects/semif`):
тот же system-промпт, тот же JSON `{evidence, criterion, options}` в user, тот же
`enable_thinking=false`. Ответ читается не из текста, а из логитов буквы варианта —
как в `direct.py` самого SemIf, только через HTTP-сервер llama.cpp (`decide.sh`
поднимает его на 127.0.0.1:8095).

    decide.py --question "Which workspace?" --options "код,медиа,игры" --state-file tabs.txt
    cat tabs.txt | decide.py --options-file spaces.txt --question "Which workspace?"
    DECIDE_RAW=1 decide.py ...        # JSON со всеми вероятностями и временем

Порядок вариантов фиксирован и влияет на ответ: в JevBench у одной из моделей
разворот вариантов уронил точность с 72 % до 21 %. Не переставляй без замера.

Вероятности — условные счёты вариантов, а не калиброванная уверенность решения
(формулировка авторов SemIf; ECE у этого класса измеряется отдельно).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SEMIF_SRC = os.environ.get("SEMIF_SRC", str(Path.home() / "Projects/semif/src"))
PORT = os.environ.get("DECIDER_PORT", "8095")
TOP_K = int(os.environ.get("DECIDE_TOPK", "20"))
TIMEOUT = float(os.environ.get("DECIDE_TIMEOUT", "120"))

sys.path.insert(0, SEMIF_SRC)
try:
    from semif_phase1.core import LETTERS, direct_messages  # noqa: E402
except ImportError as error:  # pragma: no cover - подсказка вместо трассировки
    sys.exit(f"не нашёл харнесс SemIf в {SEMIF_SRC}: {error}\n"
             f"он задаёт формат запроса — без него decide.py не работает")


def read_text(value: str | None, path: str | None, label: str) -> str | None:
    if path:
        return Path(path).expanduser().read_text(encoding="utf-8")
    return value


def parse_state(raw: str) -> object:
    """JSON-состояние оставляем структурой: так его и ждёт validate_row."""
    text = raw.strip()
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw


def softmax(values: dict[str, float]) -> dict[str, float]:
    peak = max(values.values())
    weights = {letter: math.exp(value - peak) for letter, value in values.items()}
    total = sum(weights.values())
    return {letter: weight / total for letter, weight in weights.items()}


def readout(completion: dict, letters: str) -> tuple[str, dict[str, float], list[str]]:
    """Вероятности букв из логитов первого токена, ограниченные объявленными вариантами."""
    position = completion["choices"][0].get("logprobs", {}).get("content") or []
    if not position:
        raise RuntimeError("сервер не вернул logprobs — проверь, что он поднят с --jinja и без --no-logprobs")
    first = position[0]
    seen: dict[str, float] = {}
    for item in [first, *(first.get("top_logprobs") or [])]:
        token = (item.get("token") or "").strip()
        # `"" in "ABC…"` в Python истинно, поэтому пустой токен проверяем отдельно:
        # иначе он занимает место среди вариантов и забирает часть вероятности.
        if token and token in letters and token not in seen:
            seen[token] = float(item["logprob"])

    chosen = (first.get("token") or "").strip()
    if not chosen or chosen not in letters:
        raise RuntimeError(
            f"модель ответила не буквой варианта: {chosen!r} "
            f"(первые 80 символов: {json.dumps(position[:3], ensure_ascii=False)[:120]}). "
            "Обычно это значит, что шаблон включил <think> — сервер поднят с enable_thinking=false?"
        )
    missing = [letter for letter in letters if letter not in seen]
    probabilities = softmax(seen)
    return chosen, probabilities, missing


def ask(row: dict, letters: str) -> dict:
    body = {
        "model": "local",
        "messages": direct_messages(row),
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": TOP_K,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            completion = json.load(response)
    except urllib.error.URLError as error:
        sys.exit(f"решатель не отвечает на 127.0.0.1:{PORT} ({error.reason}) — запусти ./start.sh decider")
    elapsed = time.perf_counter() - started

    chosen, probabilities, missing = readout(completion, letters)
    return {
        "letter": chosen,
        "probabilities": probabilities,
        "missing_from_topk": missing,
        "input_tokens": completion.get("usage", {}).get("prompt_tokens"),
        "seconds": round(elapsed, 3),
        "prompt_sha256": hashlib.sha256(json.dumps(direct_messages(row)).encode()).hexdigest()[:16],
        "readout": f"softmax over top-{TOP_K} first-token logprobs restricted to declared letters",
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True, description="Локальный решатель Jev-класса.")
    parser.add_argument("--question", help="критерий выбора (строка)")
    parser.add_argument("--question-file", help="то же из файла")
    parser.add_argument("--options", help="варианты через запятую, по порядку")
    parser.add_argument("--options-file", help="то же: по одному варианту на строку")
    parser.add_argument("--state", help="состояние: текст или JSON")
    parser.add_argument("--state-file", help="состояние из файла (JSON или текст)")
    parser.add_argument("--id", default="cli", help="идентификатор решения (попадает в вывод)")
    args = parser.parse_args()

    question = read_text(args.question, args.question_file, "критерий")
    state = read_text(args.state, args.state_file, "состояние")
    if state is None and not sys.stdin.isatty():
        state = sys.stdin.read()
    if args.options_file:
        options = [line.strip() for line in Path(args.options_file).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.options:
        options = [item.strip() for item in args.options.split(",") if item.strip()]
    else:
        options = []

    if not question:
        sys.exit("нет критерия: --question или --question-file")
    if not state or not str(state).strip():
        sys.exit("нет состояния: --state, --state-file или stdin")
    if not 2 <= len(options) <= len(LETTERS):
        sys.exit(f"вариантов должно быть 2…{len(LETTERS)} (сейчас {len(options)})")

    row = {
        "id": args.id,
        "state": parse_state(state) if isinstance(state, str) else state,
        "question": question,
        "options": [{"id": option, "description": option} for option in options],
    }
    letters = LETTERS[: len(options)]
    result = ask(row, letters)
    chosen = result["letter"]
    index = letters.index(chosen)
    probability = result["probabilities"][chosen]

    if result["missing_from_topk"]:
        print(f"! вариантов нет в top-{TOP_K} логитов: {', '.join(result['missing_from_topk'])}", file=sys.stderr)

    if os.environ.get("DECIDE_RAW"):
        print(json.dumps({**result, "option": options[index], "options": options, "question": question}, ensure_ascii=False, indent=2))
        return 0

    print(f"ANSWER: {chosen} — {options[index]} (P {probability:.2f})")
    spread = " · ".join(f"{letter}={option} {result['probabilities'].get(letter, 0):.2f}"
                        for letter, option in zip(letters, options))
    print(f"PROBS: {spread}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
