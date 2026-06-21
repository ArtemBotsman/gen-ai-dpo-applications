"""Генерация синтетических персон по демографическим якорям."""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import get_model, make_client
from schema import DemProfile, Persona

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "dem_profiles.json"


def load_dem_profiles() -> list[DemProfile]:
    raw = json.loads(INPUT.read_text(encoding="utf-8"))
    return [DemProfile.model_validate(x) for x in raw]


def generate_personas(profiles: list[DemProfile] | None = None) -> list[Persona]:
    profiles = profiles or load_dem_profiles()
    client = make_client()
    personas: list[Persona] = []

    for prof in profiles:
        p = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Сгенерируй одну правдоподобную российскую персону для социологического "
                        "эксперимента. Это СИНТЕТИЧЕСКИЕ данные. Не используй известных людей. "
                        "Поля age, gender, education, income_band, city_size, region — "
                        "строго как в карточке. Поле id — как в карточке."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Карточка:\n{prof.model_dump_json(indent=2)}\n\n"
                        f"id персоны: {prof.id}. Придумай name, bio (2 предложения), "
                        "fairness_note — как человек понимает справедливый делёж денег."
                    ),
                },
            ],
            response_model=Persona,
            temperature=0.8,
            max_retries=2,
        )
        p.id = prof.id
        personas.append(p)

    return personas


if __name__ == "__main__":
    ps = generate_personas()
    for p in ps:
        print(f"{p.id}: {p.name}, {p.age}, {p.region} — {p.fairness_note[:60]}...")
