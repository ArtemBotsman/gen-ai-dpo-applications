"""
Главный конвейер «Лаборатория справедливости».

Запуск:
    python pipeline.py           # полный прогон
    python pipeline.py --quick   # 3 персоны, 1 раунд фокус-группы
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from analyze import check_ghost_quotes, extract_aspects, judge_summary
from focus_group import run_focus_group
from mock_data import (
    mock_aspects,
    mock_ghost,
    mock_judge,
    mock_personas,
    mock_survey,
    mock_transcript,
    mock_ultimatum,
)
from personas import generate_personas, load_dem_profiles
from survey import run_survey
from ultimatum import run_ultimatum

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
INPUT = ROOT / "input"


@dataclass
class RunStats:
    llm_calls: int = 0
    phases: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)

    def bump(self, n: int = 1) -> None:
        self.llm_calls += n


def _save(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json")
    elif isinstance(obj, list) and obj and hasattr(obj[0], "model_dump"):
        data = [x.model_dump(mode="json") for x in obj]
    else:
        data = obj
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run(*, quick: bool = False, mock: bool = False) -> dict:
    stats = RunStats()
    profiles = load_dem_profiles()
    if quick:
        profiles = profiles[:3]

    if mock:
        print("=== Режим --mock (без LLM) ===")
        personas = mock_personas()[: len(profiles)]
        transcript = mock_transcript(personas)
        aspects = mock_aspects()
        ghost = mock_ghost()
        judge = mock_judge()
        survey = mock_survey()[: len(personas)]
        ult = mock_ultimatum()
        ult = [r for r in ult if r.persona_id in {p.id for p in personas}]
        stats.llm_calls = 0
        _save("personas.json", personas)
        _save("focus_transcript.json", transcript)
        _save("transcript_aspects.json", aspects)
        _save("ghost_quotes.json", ghost)
        _save("judge_report.json", judge)
        _save("survey_answers.json", survey)
        _save("ultimatum_rounds.json", ult)
        summary = {
            "llm_calls": 0,
            "total_sec": 0.1,
            "phases": {"mock": True},
            "personas": len(personas),
            "ultimatum_rounds": len(ult),
            "ghost_quote_rate": ghost.ghost_rate,
            "judge_score": judge.overall_score,
        }
        _save("run_summary.json", summary)
        print(f"  {len(personas)} персон, {len(ult)} ultimatum-раундов")
        print(f"Артефакты: {OUT}/")
        return summary

    print("=== Фаза 0: персоны ===")
    t0 = time.perf_counter()
    personas = generate_personas(profiles)
    stats.llm_calls += len(personas)
    stats.phases["personas_sec"] = round(time.perf_counter() - t0, 1)
    _save("personas.json", personas)
    print(f"  {len(personas)} персон, {stats.phases['personas_sec']}с")

    print("\n=== Фаза 1: фокус-группа ===")
    t0 = time.perf_counter()
    transcript = run_focus_group(personas, rounds=1 if quick else 2)
    stats.llm_calls += len(transcript.utterances)
    stats.phases["focus_group_sec"] = round(time.perf_counter() - t0, 1)
    _save("focus_transcript.json", transcript)
    print(f"  {len(transcript.utterances)} реплик")

    print("\n=== Анализ транскрипта ===")
    t0 = time.perf_counter()
    aspects = extract_aspects(transcript)
    stats.llm_calls += len(aspects)
    ghost = check_ghost_quotes(transcript, aspects)
    judge = judge_summary(transcript, ghost)
    stats.llm_calls += 1
    stats.phases["analyze_sec"] = round(time.perf_counter() - t0, 1)
    _save("transcript_aspects.json", aspects)
    _save("ghost_quotes.json", ghost)
    _save("judge_report.json", judge)
    print(f"  ghost-цитат: {ghost.ghost_quotes}/{ghost.total_quotes} ({ghost.ghost_rate:.0%})")

    print("\n=== Фаза 2: анкета (silicon sampling) ===")
    t0 = time.perf_counter()
    survey = run_survey(personas)
    stats.llm_calls += len(survey)
    stats.phases["survey_sec"] = round(time.perf_counter() - t0, 1)
    _save("survey_answers.json", survey)
    print(f"  {len(survey)} анкет")

    print("\n=== Фаза 3: ultimatum ===")
    t0 = time.perf_counter()
    offers = [10, 20, 30] if quick else [10, 20, 30, 40, 50]
    ult = run_ultimatum(personas, offers=offers)
    stats.llm_calls += len(ult)
    stats.phases["ultimatum_sec"] = round(time.perf_counter() - t0, 1)
    _save("ultimatum_rounds.json", ult)
    resp = [r for r in ult if r.role == "responder"]
    acc = sum(1 for r in resp if r.accept) / len(resp) if resp else 0
    print(f"  {len(ult)} раундов, доля принятий (responder): {acc:.0%}")

    total_sec = round(time.perf_counter() - stats.started_at, 1)
    summary = {
        "llm_calls": stats.llm_calls,
        "total_sec": total_sec,
        "phases": stats.phases,
        "personas": len(personas),
        "ultimatum_rounds": len(ult),
        "ghost_quote_rate": ghost.ghost_rate,
        "judge_score": judge.overall_score,
    }
    _save("run_summary.json", summary)
    print(f"\n=== Готово: {stats.llm_calls} LLM-вызовов за {total_sec}с ===")
    print(f"Артефакты: {OUT}/")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="быстрый прогон для отладки")
    ap.add_argument("--mock", action="store_true", help="демо без LLM")
    args = ap.parse_args()
    run(quick=args.quick, mock=args.mock)


if __name__ == "__main__":
    main()
