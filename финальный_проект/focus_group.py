"""Фаза 1: симуляция фокус-группы про справедливый делёж."""

from __future__ import annotations

from datetime import date

from llm_client import get_model, make_client
from schema import FocusGroupTranscript, FocusUtterance, Persona


TOPIC = "Как, по-вашему, честно разделить 100 ₽ между двумя незнакомцами?"


def run_focus_group(personas: list[Persona], rounds: int = 2) -> FocusGroupTranscript:
    client = make_client()
    utterances: list[FocusUtterance] = []
    history = ""

    for rnd in range(1, rounds + 1):
        for persona in personas:
            u = client.chat.completions.create(
                model=get_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Ты участник фокус-группы. Персона:\n"
                            f"Имя: {persona.name}, {persona.age} лет, {persona.region}, "
                            f"{persona.education}, доход {persona.income_band}.\n"
                            f"О справедливости: {persona.fairness_note}\n"
                            f"Био: {persona.bio}\n"
                            "Ответь одной репликой 1–3 предложения, по-человечески, "
                            "можно согласиться или поспорить с другими."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Тема: {TOPIC}\n"
                            f"Раунд {rnd}.\n"
                            f"Предыдущие реплики:\n{history or '(пока никто не говорил)'}\n"
                            "Твоя реплика:"
                        ),
                    },
                ],
                response_model=FocusUtterance,
                temperature=0.7,
                max_retries=2,
            )
            u.speaker_id = persona.id
            u.speaker_name = persona.name
            utterances.append(u)
            history += f"\n{persona.name}: {u.text}"

    return FocusGroupTranscript(
        topic=TOPIC,
        utterances=utterances,
        generated_at=date.today(),
    )
