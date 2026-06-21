"""
Eval финального проекта: ≥15 кейсов, правильность + путь.

Запуск (после pipeline.py):
    python eval.py
    python eval.py --quick   # без повторных LLM-вызовов, только по output/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
INPUT = ROOT / "input"


def _load(name: str):
    p = OUT / name
    if not p.exists():
        raise FileNotFoundError(f"Нет {p}. Сначала: python pipeline.py")
    return json.loads(p.read_text(encoding="utf-8"))


def eval_ultimatum_responder(rounds: list[dict], baseline: dict) -> list[dict]:
    """15+ кейсов: offer → accept vs human baseline."""
    acc_map = {int(k): float(v) for k, v in baseline["acceptance_by_offer"].items()}
    cases = []
    for r in rounds:
        if r["role"] != "responder":
            continue
        offer = int(r["offer"])
        if offer not in acc_map:
            continue
        human_accept_rate = acc_map[offer]
        # ожидаем: при offer≤20 чаще отказ, при ≥40 чаще принятие
        expected_accept = human_accept_rate >= 0.5
        actual = bool(r["accept"])
        direction_ok = actual == expected_accept or offer == 30  # пограничный
        cases.append(
            {
                "id": r["round_id"],
                "type": "ultimatum_responder",
                "offer": offer,
                "accept": actual,
                "human_accept_rate": human_accept_rate,
                "ok": direction_ok,
                "path": {"role": "responder", "persona": r["persona_id"]},
            }
        )
    return cases


def eval_survey(survey: list[dict], ess: dict) -> list[dict]:
    cases = []
    cohort_map = ess["cohort_map"]
    by_cohort = ess["by_cohort"]
    for ans in survey:
        pid = ans["persona_id"]
        cohort = cohort_map.get(pid)
        if not cohort:
            continue
        base = by_cohort[cohort]
        for field in ("trust_people", "trust_strangers", "risk_appetite"):
            synth = ans[field]
            real = base[field]
            delta = abs(synth - real)
            cases.append(
                {
                    "id": f"survey_{pid}_{field}",
                    "type": "survey_compare",
                    "field": field,
                    "synthetic": synth,
                    "ess_baseline": real,
                    "delta": round(delta, 2),
                    "ok": delta <= 2.0,
                    "path": {"phase": "survey", "persona": pid},
                }
            )
    return cases


def eval_ghost(ghost: dict) -> list[dict]:
    return [
        {
            "id": "ghost_quotes",
            "type": "hallucination_check",
            "ghost_rate": ghost["ghost_rate"],
            "ghost_quotes": ghost["ghost_quotes"],
            "total": ghost["total_quotes"],
            "ok": ghost["ghost_rate"] <= 0.25,
            "path": {"phase": "analyze", "technique": "ghost_quote_validator"},
        }
    ]


def eval_proposer(rounds: list[dict]) -> list[dict]:
    cases = []
    for r in rounds:
        if r["role"] != "proposer":
            continue
        offer = int(r["offer"])
        # люди в среднем предлагают ~40
        ok = 20 <= offer <= 60
        cases.append(
            {
                "id": r["round_id"],
                "type": "ultimatum_proposer",
                "offer": offer,
                "ok": ok,
                "path": {"role": "proposer", "persona": r["persona_id"]},
            }
        )
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    baseline = json.loads((INPUT / "human_baseline.json").read_text(encoding="utf-8"))
    ess = json.loads((INPUT / "ess_survey_baseline.json").read_text(encoding="utf-8"))

    ult = _load("ultimatum_rounds.json")
    survey = _load("survey_answers.json")
    ghost = _load("ghost_quotes.json")
    summary = _load("run_summary.json") if (OUT / "run_summary.json").exists() else {}

    cases: list[dict] = []
    cases += eval_ultimatum_responder(ult, baseline)
    cases += eval_survey(survey, ess)
    cases += eval_ghost(ghost)
    cases += eval_proposer(ult)

    passed = sum(1 for c in cases if c["ok"])
    total = len(cases)
    pass_rate = passed / total if total else 0

    by_type: dict[str, list] = defaultdict(list)
    for c in cases:
        by_type[c["type"]].append(c)

    print(f"Eval: {passed}/{total} pass ({pass_rate:.0%}), кейсов ≥15: {total >= 15}\n")
    for t, items in sorted(by_type.items()):
        p = sum(1 for x in items if x["ok"])
        print(f"  {t}: {p}/{len(items)}")

    if summary:
        print(f"\nПуть прогона: {summary.get('llm_calls')} LLM-вызовов, "
              f"{summary.get('total_sec')}с, фазы: {summary.get('phases')}")

    result = {
        "passed": passed,
        "total": total,
        "pass_rate": round(pass_rate, 3),
        "meets_min_15": total >= 15,
        "cases": cases,
        "summary_path": summary,
    }
    out_path = OUT / "eval_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
