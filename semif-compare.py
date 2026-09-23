#!/usr/bin/env python3
"""Сравнить предсказания SemIf с золотом и с закоммиченными предсказаниями авторов.

usage: semif-compare.py GOLD.jsonl OURS.jsonl [REFERENCE.jsonl]

Метрики совпадают с определением авторов: accuracy по всем строкам и
mean family balanced accuracy (внутри семейства — средний recall по классам,
затем среднее по семействам). Строки без предсказания не считаются решёнными,
но входят в знаменатель accuracy — как в их отчёте.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def chosen(row: dict | None) -> str | None:
    if not row:
        return None
    probs, ids = row.get("probabilities"), row.get("option_ids")
    if not probs or not ids or len(probs) != len(ids):
        return None
    return ids[max(range(len(probs)), key=lambda i: probs[i])]


def metrics(rows: list[tuple[str, str | None, str]]) -> tuple[float, float]:
    accuracy = sum(1 for gold, pred, _ in rows if gold == pred) / len(rows)
    by_family: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for gold, pred, family in rows:
        by_family[family].append((gold, pred))
    family_scores = []
    for family_rows in by_family.values():
        per_class = []
        for label in {gold for gold, _ in family_rows}:
            total = sum(1 for gold, _ in family_rows if gold == label)
            hit = sum(1 for gold, pred in family_rows if gold == label and gold == pred)
            per_class.append(hit / total if total else 0.0)
        family_scores.append(sum(per_class) / len(per_class))
    return accuracy, sum(family_scores) / len(family_scores)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    gold_rows = load(sys.argv[1])
    ours = load(sys.argv[2])
    reference = load(sys.argv[3]) if len(sys.argv) > 3 else {}

    table = []
    for rid, grow in gold_rows.items():
        gold = grow["options"][grow["label"]]["id"]
        table.append((gold, chosen(ours.get(rid)), grow.get("family", "?")))

    resolved = [t for t in table if t[1] is not None]
    accuracy, balanced = metrics(table)

    print(f"строк в золоте       : {len(table)}")
    print(f"получено решений     : {len(resolved)} ({100 * len(resolved) / len(table):.1f} %)")
    print(f"accuracy             : {accuracy:.4f}   (в знаменателе все строки)")
    print(f"mean family balanced : {balanced:.4f}")
    if resolved and len(resolved) != len(table):
        print(f"  ...по решённым     : {metrics(resolved)[1]:.4f}")

    if reference:
        same = diff = 0
        for rid, grow in gold_rows.items():
            ours_c, ref_c = chosen(ours.get(rid)), chosen(reference.get(rid))
            if ours_c is None or ref_c is None:
                continue
            same += ours_c == ref_c
            diff += ours_c != ref_c
        total = same + diff
        print(f"согласие с авторами  : {100 * same / total:.1f} % ({same}/{total})" if total else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
