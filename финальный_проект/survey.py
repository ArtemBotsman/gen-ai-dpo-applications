"""Фаза 2: silicon sampling — мини-анкета по персонам."""

from __future__ import annotations

from llm_client import get_model, make_client
from schema import Persona, SurveyAnswer


def run_survey(personas: list[Persona]) -> list[SurveyAnswer]:
    client = make_client()
    answers: list[SurveyAnswer] = []

    for persona in personas:
        ans = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты отвечаешь на анкету от лица указанной персоны. "
                        "Шкала 1–5, где 1 — «совсем нет», 5 — «полностью согласен». "
                        "min_acceptable_share — минимум ₽ из 100, который можно предложить "
                        "второму в игре ultimatum, иначе это несправедливо."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Персона: {persona.name}, {persona.age} лет, {persona.region}, "
                        f"доход {persona.income_band}. {persona.fairness_note}\n"
                        "Ответь на все поля анкеты."
                    ),
                },
            ],
            response_model=SurveyAnswer,
            temperature=0.3,
            max_retries=2,
        )
        ans.persona_id = persona.id
        answers.append(ans)

    return answers
