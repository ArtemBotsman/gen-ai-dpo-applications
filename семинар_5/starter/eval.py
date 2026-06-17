"""
Мини-оценка: 10 вопросов, проверяем:
1. Что агент завершает работу за разумное число шагов.
2. Что в трассе шагов есть ожидаемые инструменты.
3. Что в финальном ответе упомянуты ожидаемые ключевые слова (опционально).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

TRACE_PATH = Path(__file__).resolve().parent / "trace.jsonl"

CASES = [
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],
        "comment": "Базовый тест — один инструмент, одно число.",
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова одного инструмента с разными аргументами.",
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Три разных инструмента + арифметика.",
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Формула 72 / ставка = годы.",
    },
    {
        "id": 5,
        "query": "Во сколько раз вырос курс USD с января 2022 по апрель 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "Новый инструмент compare_periods — ratio курса доллара.",
    },
    {
        "id": 6,
        "query": "На сколько процентных пунктов изменилась ключевая ставка с марта 2022 по июнь 2024?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "compare_periods по key_rate, delta в п.п.",
    },
    {
        "id": 7,
        "query": "Какая ключевая ставка была весной 2022 года?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],
        "comment": "Трудный: «весна 2022» неоднозначна — март, апрель или май; агент может выбрать не тот месяц.",
    },
    {
        "id": 8,
        "query": "Что сейчас выше: ключевая ставка или индекс нищеты (инфляция плюс безработица)?",
        "expected_tools": ["get_key_rate", "get_inflation", "get_unemployment", "calculate"],
        "must_have": [],
        "comment": "Трудный: нужно собрать три метрики, сложить две, сравнить с третьей — легко перепутать порядок.",
    },
    {
        "id": 9,
        "query": "Стоит ли сейчас фиксировать ипотеку при текущей ключевой ставке и инфляции?",
        "expected_tools": ["get_key_rate", "get_inflation"],
        "must_have": [],
        "comment": "Реальный вопрос: ставка + инфляция для решения о кредите.",
    },
    {
        "id": 10,
        "query": "Как изменился курс юаня к рублю за 2024 год — с января по декабрь?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "Реальный вопрос про CNY; compare_periods fx_CNY 2024-01 vs 2024-12.",
    },
]


def run_case(
    case: dict,
    *,
    use_cache: bool = False,
    track_cost: bool = False,
    trace_path: Path | None = TRACE_PATH,
) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    res = run_agent(
        case["query"],
        max_iter=8,
        verbose=True,
        use_cache=use_cache,
        track_cost=track_cost,
        trace_path=trace_path,
    )
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    tool_match = all(t in used_tools for t in case["expected_tools"])
    text_match = all(s.lower() in answer.lower() for s in case["must_have"])
    ok = bool(answer) and tool_match and text_match and "error" not in res

    print(f"\n  tools used : {used_tools}")
    print(
        f"  expected    : {case['expected_tools']}  → {'OK' if tool_match else 'MISS'}"
    )
    print(f"  answer      : {answer[:200]}")
    print(f"  must_have   : {case['must_have']}  → {'OK' if text_match else 'MISS'}")
    print(f"  verdict     : {'PASS' if ok else 'FAIL'}")

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
        "comment": case.get("comment", ""),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Мини-оценка макро-агента")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Блок 9: общий кэш инструментов на все вопросы",
    )
    ap.add_argument(
        "--cost",
        action="store_true",
        help="Блок 10: показать токены и стоимость по шагам",
    )
    ap.add_argument(
        "--trace",
        type=Path,
        default=TRACE_PATH,
        help="JSONL-лог шагов",
    )
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    results = [
        run_case(c, use_cache=a.cache, track_cost=a.cost, trace_path=a.trace)
        for c in CASES
    ]
    passed = sum(1 for r in results if r["ok"])

    print(f"\n{'=' * 70}\nИтого: {passed}/{len(CASES)} пройдено")
    for r in results:
        mark = "[OK]  " if r["ok"] else "[FAIL]"
        print(f"  {mark} Q{r['id']} ({r['steps']} шагов) — {r['query'][:60]}")

    if a.cache:
        h, m = CACHE_STATS["hits"], CACHE_STATS["misses"]
        print(
            f"\n[кэш] на {len(CASES)} вопросах: {h} попаданий из {h + m} обращений "
            f"к инструментам."
        )

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")
    if a.trace:
        print(f"Трасса: {a.trace}")


if __name__ == "__main__":
    main()
