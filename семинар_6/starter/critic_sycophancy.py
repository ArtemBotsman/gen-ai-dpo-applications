"""
Замер «угодливости» Критика: temperature 0.0 vs 0.7 на заведомо битых ответах.

Запуск:
    python critic_sycophancy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer

FAKE_BROKEN = [
    {
        "name": "арифметика без calculate",
        "question": "На сколько EUR дороже USD сегодня?",
        "plan": Plan(
            reasoning="Сравнить курсы",
            subquestions=[
                SubQuestion(id=1, question="курс USD?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="курс EUR?", expected_tools=["get_fx_rate"]),
                SubQuestion(
                    id=3,
                    question="разница EUR-USD",
                    expected_tools=["calculate"],
                    depends_on=[1, 2],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="USD",
                answer="USD=82.5 руб.",
                used_tools=["get_fx_rate"],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="EUR",
                answer="EUR=89.0 руб.",
                used_tools=["get_fx_rate"],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="разница",
                answer="EUR дороже USD на 6.5 руб.",
                used_tools=[],  # нет calculate!
            ),
        },
    },
    {
        "name": "выдуманное число",
        "question": "Какая ключевая ставка?",
        "plan": Plan(
            reasoning="Узнать ставку",
            subquestions=[
                SubQuestion(id=1, question="ставка", expected_tools=["get_key_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="ставка",
                answer="Ключевая ставка 42% годовых.",
                used_tools=["get_key_rate"],
            ),
        },
    },
    {
        "name": "несогласованные данные",
        "question": "Во сколько раз USD вырос?",
        "plan": Plan(
            reasoning="Два курса и отношение",
            subquestions=[
                SubQuestion(id=1, question="USD 2022", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="USD сегодня", expected_tools=["get_fx_rate"]),
                SubQuestion(
                    id=3,
                    question="отношение",
                    expected_tools=["calculate"],
                    depends_on=[1, 2],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="2022",
                answer="USD=76.2",
                used_tools=["get_fx_rate"],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="сегодня",
                answer="USD=82.5",
                used_tools=["get_fx_rate"],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="ratio",
                answer="USD вырос в 2.5 раза",
                used_tools=["calculate"],
            ),  # 82.5/76.2 ≠ 2.5
        },
    },
    {
        "name": "ответ с ошибкой инструмента",
        "question": "Инфляция за март 2024?",
        "plan": Plan(
            reasoning="ИПЦ",
            subquestions=[
                SubQuestion(id=1, question="ИПЦ", expected_tools=["get_inflation"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="ИПЦ",
                answer="(ошибка: нет данных)",
                used_tools=[],
            ),
        },
    },
    {
        "name": "план не покрывает вопрос",
        "question": "Реальная ставка = номинальная минус инфляция?",
        "plan": Plan(
            reasoning="Только ставка",
            subquestions=[
                SubQuestion(id=1, question="ставка", expected_tools=["get_key_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="ставка",
                answer="16%",
                used_tools=["get_key_rate"],
            ),
        },
    },
]

RUNS = 10


def measure_case(case: dict) -> dict:
    false_accept = {0.0: 0, 0.7: 0}
    for temp in (0.0, 0.7):
        for _ in range(RUNS):
            v = critic(
                case["question"],
                case["plan"],
                case["answers"],
                temperature=temp,
            )
            if v.ok:
                false_accept[temp] += 1
    return {
        "name": case["name"],
        "false_accept_t0": false_accept[0.0],
        "false_accept_t07": false_accept[0.7],
        "runs": RUNS,
    }


def main():
    print(f"Замер угодливости Критика: {RUNS} прогонов × T=0.0 и T=0.7\n")
    print(f"{'Битый кейс':<28} | T=0.0 | T=0.7")
    print("-" * 50)
    results = []
    for case in FAKE_BROKEN:
        r = measure_case(case)
        results.append(r)
        print(
            f"{r['name']:<28} | {r['false_accept_t0']}/{RUNS}   | {r['false_accept_t07']}/{RUNS}"
        )

    out = Path(__file__).parent / "critic_sycophancy_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
