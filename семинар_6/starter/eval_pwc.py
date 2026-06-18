"""
Eval мульти-агента: 6 вопросов × 3 конфигурации.

Конфигурации:
  1) одиночный агент С5
  2) PWC без валидатора
  3) PWC + валидатор схемы

Запуск:
    python eval_pwc.py           # N=5 (по умолчанию)
    python eval_pwc.py -n 3      # быстрее
    python eval_pwc.py --single  # N=1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc

CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": "Ошибка C: одиночный считает в уме. PWC + calculate. Параллельность: 2 независимых get_fx_rate.",
        "must_have_keywords": ["раз", "usd"],
        "parallel_bench": True,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": "Ошибка B: нужен поиск последнего месяца ИПЦ.",
        "must_have_keywords": ["%"],
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Ошибка D: Планировщик галлюцинирует get_cumulative_inflation. "
            "Валидатор должен поймать и перепланировать."
        ),
        "must_have_keywords": ["%"],
        "validator_fix": True,
    },
    {
        "id": "Q4",
        "query": (
            "Сравни официальные курсы USD, EUR и CNY к рублю на сегодня — "
            "какая валюта самая дорогая?"
        ),
        "comment": "3+ независимых подвопроса — замер параллельности.",
        "must_have_keywords": [],
        "parallel_bench": True,
    },
    {
        "id": "Q5",
        "query": (
            "Если я положу деньги на вклад под текущую ключевую ставку на год, "
            "опережу ли я инфляцию? Посчитай реальную доходность."
        ),
        "comment": "Реальный вопрос: ставка + ИПЦ + calculate.",
        "must_have_keywords": ["%"],
    },
    {
        "id": "Q6",
        "query": (
            "Насколько ключевая ставка ЦБ выше инфляции г/г за март 2026 — "
            "в процентных пунктах?"
        ),
        "comment": "Простой multi-hop; валидатор ловит выдуманные tools в плане.",
        "must_have_keywords": ["п.", "%"],
        "validator_fix": True,
    },
]

VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _check_single(case: dict, result: dict) -> dict:
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    needs_calc = case["id"] in {"Q1", "Q5", "Q6"}
    arith_without_calc = needs_calc and "calculate" not in used and bool(ans)
    ok = bool(ans) and not hallucinated and must and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _check_pwc(case: dict, result: dict, *, require_no_plan_hallucination: bool) -> dict:
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES

    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])

    ok = bool(result.get("answer")) and not hallucinated and must
    if require_no_plan_hallucination:
        ok = ok and not plan_hallucinated

    # Q3: успех если нет галлюцинаций в плане ИЛИ честный пустой план с объяснением
    if case["id"] == "Q3" and plan is not None and not plan.subquestions:
        ok = bool(plan.reasoning) and "не" in plan.reasoning.lower()

    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:180],
    }


def run_case(case: dict, *, n: int = 5) -> dict:
    single = {"runs": [], "pass": 0}
    pwc = {"runs": [], "pass": 0}
    pwc_val = {"runs": [], "pass": 0}

    for _ in range(n):
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": str(e), "trace": []}
        c1 = _check_single(case, r1)
        single["runs"].append(c1)
        single["pass"] += int(c1["ok"])

        try:
            r2 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=False)
        except Exception as e:
            r2 = {"answer": None, "error": str(e), "trace": [], "plan": None}
        c2 = _check_pwc(case, r2, require_no_plan_hallucination=False)
        pwc["runs"].append(c2)
        pwc["pass"] += int(c2["ok"])

        try:
            r3 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=True)
        except Exception as e:
            r3 = {"answer": None, "error": str(e), "trace": [], "plan": None}
        c3 = _check_pwc(case, r3, require_no_plan_hallucination=True)
        pwc_val["runs"].append(c3)
        pwc_val["pass"] += int(c3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc": pwc,
        "pwc_validator": pwc_val,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="N=1")
    ap.add_argument("-n", type=int, default=5, help="прогонов на кейс (default=5)")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    print(f"Eval С6: {len(CASES)} кейсов × 3 конфигурации × N={n}\n")
    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:65]}...")
        r = run_case(case, n=n)
        results.append(r)
        print(
            f"   single: {r['single']['pass']}/{n}  "
            f"pwc: {r['pwc']['pass']}/{n}  "
            f"pwc+val: {r['pwc_validator']['pass']}/{n}"
        )
        for run in r["pwc"]["runs"][:1]:
            if run.get("hallucinated_in_plan"):
                print(f"   ⚠ pwc план: {run['hallucinated_in_plan']}")
        print()

    print("=" * 60)
    print("ИТОГО (доля pass):")
    for r in results:
        print(
            f"  {r['id']}: single {r['single']['pass']}/{n}  "
            f"pwc {r['pwc']['pass']}/{n}  "
            f"pwc+val {r['pwc_validator']['pass']}/{n}"
        )

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
