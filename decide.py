#!/usr/bin/env python3
"""decide.py — Jev-класс локально: состояние + критерий + варианты → буква и вероятности.

Формат запроса и чтение ответа не выдуманы заново, а взяты у авторов чекпойнта
(`Mapika/decider-4b` v2.1): `decider/prompt.py::build` — раскладка state-first, plain
(без chat-шаблона):

    Context:
    <состояние>

    Question: <критерий>
    Options:
    (A) <вариант>
    (B) <вариант>
    Answer: (

Ответ читается не из текста, а из логитов буквы: llama-server отдаёт логиты в позиции
последнего токена промпта (`" ("`), скрипт берёт их из `/completion` и делит на
температуру из `decider_config.json` авторов (для Choice — 1.11). Это ровно то, что
делает их `decide_gguf.py`; никакого текста модель не порождает.

    decide.py --question "Which workspace?" --options "код,медиа,игры" --state-file tabs.txt
    cat tabs.txt | decide.py --options-file spaces.txt --question "Which workspace?"
    DECIDE_RAW=1 decide.py ...        # JSON со всеми вероятностями и временем

Типы ответов авторов (decider/infer.py → _schema_to_questions): промпт у всех один, тип
меняет только температуру (choice 1.11, noul 1.56, score 1.287 из temperature_by_type) и
набор вариантов: noul — всегда ["no", "yes"]; score — метки уровней verbatim по возрастанию:

    decide.py --type noul --question "Is the checksum valid?" --state-file build.log --json
    decide.py --type score --question "How severe?" --options "0: none,1: low,2: medium,3: high,4: critical" --state incident.txt

Почему `/completion`, а не `/v1/chat/completions`: у чекпойнта в конфиге
`"layout": "plain"`, он не чат-модель. Chat-шаблон Qwen3.5 добавил бы `<|im_start|>` и
сломал бы раскладку — первым токеном стала бы не буква.

Порядок вариантов фиксирован и влияет на ответ: в JevBench у одной из моделей
разворот вариантов уронил точность с 72 % до 21 %. Не переставляй без замера.

Вероятности — условные счёты вариантов, а не калиброванная уверенность решения.
У этого чекпойнта ECE измеряется авторами отдельно (0.031 на их regression set).
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

HERE = Path(__file__).resolve().parent

# Узкая раскладка авторов (decider.prompt.NARROW): до 10 вариантов, "(A) …". Больше —
# их «широкая» голова на 255 однотокенных меток; она нам не нужна и не реализована.
LETTERS = "ABCDEFGHIJ"
PORT = os.environ.get("DECIDER_PORT", "8095")
TOP_K = int(os.environ.get("DECIDE_TOPK", "20"))
TIMEOUT = float(os.environ.get("DECIDE_TIMEOUT", "120"))
CONFIG_PATH = Path(os.environ.get("DECIDER_CONFIG", HERE / "models" / "decider-4b-v2.1-config.json"))
# Температуры подогнаны авторами по типу ответа (decider_config.json → temperature_by_type,
# их decider.calibrate.fit_by_type), не нами: промпт у всех типов один, тип меняет только её.
FALLBACK_TEMPERATURES = {"choice": 1.11, "noul": 1.56, "score": 1.287}
REFERENCE_STATE_TOKENS = 1536        # на этом обрезает состояние их decide_gguf.py; max_state_tokens в конфиге — 32768
ANNOTATE_MIN = 8                     # decider.systemone.ANNOTATE_MIN


def read_text(value: str | None, path: str | None) -> str | None:
    if path:
        return Path(path).expanduser().read_text(encoding="utf-8")
    return value


def parse_state(raw: str) -> object:
    """JSON-состояние оставляем структурой: его сериализует render_state, как у авторов."""
    text = raw.strip()
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw


def annotate_indices(value: object, min_len: int = ANNOTATE_MIN) -> object:
    """Проставить позицию каждому элементу длинного массива — как `decider.systemone.annotate_indices`.

    Без этого путь вида `records[47].text` заставляет модель считать 47 элементов; с
    записанным индексом это поиск. Авторский замер на их пробе json_k64: 0.49 → 0.57.
    """
    if isinstance(value, list):
        if len(value) >= min_len:
            return [({"_index": i, **annotate_indices(item, min_len)} if isinstance(item, dict)
                     else {"_index": i, "value": annotate_indices(item, min_len)})
                    for i, item in enumerate(value)]
        return [annotate_indices(item, min_len) for item in value]
    if isinstance(value, dict):
        return {key: annotate_indices(item, min_len) for key, item in value.items()}
    return value


def render_state(state: object) -> str:
    """Строка состояния так, как её видит модель (decider.systemone.render_state)."""
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def build_prompt(state: object, question: str, options: list[str]) -> str:
    """Промпт в раскладке decider.prompt.build (state_first, plain, один вопрос)."""
    lines = [f"Context:\n{render_state(state)}", f"\n\nQuestion: {question}\nOptions:"]
    lines.extend(f"\n({LETTERS[index]}) {option}" for index, option in enumerate(options))
    lines.append("\nAnswer: (")
    return "".join(lines)


def temperature_for(answer_type: str) -> float:
    """Температура ответов типа answer_type из конфига авторов; таблица выше — если файла нет."""
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return FALLBACK_TEMPERATURES[answer_type]
    by_type = config.get("temperature_by_type") or {}
    try:
        return float(by_type.get(answer_type, config.get("temperature", FALLBACK_TEMPERATURES[answer_type])))
    except (TypeError, ValueError):
        return FALLBACK_TEMPERATURES[answer_type]


def readout(completion: dict, letters: str, temperature: float) -> tuple[str, dict[str, float], list[str]]:
    """Вероятности букв из логитов первого токена, ограниченные объявленными вариантами."""
    positions = completion.get("completion_probabilities") or []
    if not positions:
        raise RuntimeError("сервер не вернул логиты — /completion поднят без logprobs?")
    first = positions[0]
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
            f"(первые токены: {json.dumps(first.get('top_logprobs', [])[:4], ensure_ascii=False)[:160]}). "
            "Обычно это значит, что промпт ушёл через chat-шаблон: чекпойнт обучен на "
            "plain-раскладке и должен спрашиваться через /completion, а не /v1/chat/completions."
        )
    missing = [letter for letter in letters if letter not in seen]
    # Авторы делят логиты на температуру до softmax (decide_gguf.py: `z = lg[:n] / t`).
    # logprob = logit − logsumexp, а softmax по подмножеству инвариантен к сдвигу,
    # поэтому делить можно сами logprob.
    shifted = {letter: value / temperature for letter, value in seen.items()}
    peak = max(shifted.values())
    weights = {letter: math.exp(value - peak) for letter, value in shifted.items()}
    total = sum(weights.values())
    return chosen, {letter: weight / total for letter, weight in weights.items()}, missing


def ask(prompt: str, letters: str, temperature: float) -> dict:
    body = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0,      # выбор буквы должен быть воспроизводимым; логиты от неё не зависят
        "logprobs": TOP_K,
        "cache_prompt": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
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

    chosen, probabilities, missing = readout(completion, letters, temperature)
    tokens = completion.get("tokens_evaluated")
    return {
        "letter": chosen,
        "probabilities": probabilities,
        "missing_from_topk": missing,
        "input_tokens": tokens,
        "seconds": round(elapsed, 3),
        "temperature": temperature,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "readout": f"softmax over top-{TOP_K} first-token logits restricted to declared letters, divided by T={temperature}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True, description="Локальный решатель Jev-класса (decider-4b).")
    parser.add_argument("--question", help="критерий выбора (строка)")
    parser.add_argument("--question-file", help="то же из файла")
    parser.add_argument("--options", help="варианты через запятую, по порядку")
    parser.add_argument("--options-file", help="то же: по одному варианту на строку")
    parser.add_argument("--state", help="состояние: текст или JSON")
    parser.add_argument("--state-file", help="состояние из файла (JSON или текст)")
    parser.add_argument("--id", default="cli", help="идентификатор решения (попадает в вывод)")
    parser.add_argument("--type", choices=("choice", "noul", "score"), default="choice",
                        help="тип ответа авторов: выбор из вариантов, да/нет, уровень по меткам")
    parser.add_argument("--json", action="store_true",
                        help="один JSON-объект в stdout вместо человекочитаемого вывода")
    parser.add_argument("--bands", metavar="ASK,AUTO",
                        help="полоса уверенности для --json: human < ASK, auto ≥ AUTO, иначе ask")
    args = parser.parse_args()

    bands = None
    if args.bands:
        try:
            ask_level, auto_level = (float(part) for part in args.bands.split(","))
        except ValueError:
            sys.exit(f"--bands ожидает два числа через запятую, например 0.5,0.9 (сейчас {args.bands!r})")
        bands = (ask_level, auto_level)

    question = read_text(args.question, args.question_file)
    state = read_text(args.state, args.state_file)
    if state is None and not sys.stdin.isatty():
        state = sys.stdin.read()
    if args.type == "noul":
        # У авторов noul — это ровно ["no", "yes"] (их _schema_to_questions); переданные
        # --options игнорируются, вопрос обязан быть да/нет.
        options = ["no", "yes"]
    elif args.options_file:
        options = [line.strip() for line in Path(args.options_file).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.options:
        options = [item.strip() for item in args.options.split(",") if item.strip()]
    else:
        options = []

    if not question:
        sys.exit("нет критерия: --question или --question-file")
    if not state or not str(state).strip():
        if args.type == "noul":
            # noul — прямое да/нет по самому вопросу; у авторов такие примеры часто
            # без состояния. Пустой Context, не «null».
            state = ""
        else:
            sys.exit("нет состояния: --state, --state-file или stdin")
    if not 2 <= len(options) <= len(LETTERS):
        sys.exit(f"вариантов должно быть 2…{len(LETTERS)} (сейчас {len(options)})")

    parsed = parse_state(state) if isinstance(state, str) else state
    prompt = build_prompt(parsed, question, options)
    letters = LETTERS[: len(options)]
    temperature = temperature_for(args.type)
    result = ask(prompt, letters, temperature)
    chosen = result["letter"]
    index = letters.index(chosen)
    probability = result["probabilities"][chosen]

    if result["missing_from_topk"]:
        print(f"! вариантов нет в top-{TOP_K} логитов: {', '.join(result['missing_from_topk'])}", file=sys.stderr)
    if isinstance(result["input_tokens"], int) and result["input_tokens"] > REFERENCE_STATE_TOKENS:
        print(f"! промпт {result['input_tokens']} токенов, а decide_gguf.py авторов обрезает состояние "
              f"на {REFERENCE_STATE_TOKENS} — пример не из обученной формы", file=sys.stderr)

    if args.json:
        # Машиночитаемый договор (его ест ask_decider / jev-guard): stdout остаётся
        # чистым JSON, предупреждения выше живут на stderr.
        payload = {
            "answer": options[index],
            "letter": chosen,
            "probabilities": result["probabilities"],
            "type": args.type,
            "elapsed_s": result["seconds"],
        }
        if bands is not None:
            ask_level, auto_level = bands
            payload["band"] = "human" if probability < ask_level else "auto" if probability >= auto_level else "ask"
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if os.environ.get("DECIDE_RAW"):
        print(json.dumps({**result, "option": options[index], "options": options, "question": question},
                         ensure_ascii=False, indent=2))
        return 0

    print(f"ANSWER: {chosen} — {options[index]} (P {probability:.2f})")
    spread = " · ".join(f"{letter}={option} {result['probabilities'].get(letter, 0):.2f}"
                        for letter, option in zip(letters, options))
    print(f"PROBS: {spread}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
