"""Фаза 3: игра «ультиматум» с персонами (Horton-style agents)."""

from __future__ import annotations

from llm_client import get_model, make_client
from schema import Persona, ProposerMove, ResponderMove, UltimatumRound

DEFAULT_OFFERS = [10, 20, 30, 40, 50]


def _persona_block(p: Persona) -> str:
    return (
        f"Имя: {p.name}, {p.age} лет, {p.region}, {p.education}, "
        f"доход {p.income_band}. О справедливости: {p.fairness_note}"
    )


def propose_as(persona: Persona, round_num: int) -> ProposerMove:
    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты играешь роль Предлагающего в ultimatum game. У тебя 100 ₽. "
                    "Предложи часть второму; если откажется — оба получат 0. "
                    "Действуй как указанная персона."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Персона:\n{_persona_block(persona)}\n"
                    f"Раунд {round_num}. Сколько ₽ отдашь второму (0–100)?"
                ),
            },
        ],
        response_model=ProposerMove,
        temperature=0.7,
        max_retries=2,
    )


def respond_as(persona: Persona, offer: int) -> ResponderMove:
    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты играешь роль Отвечающего в ultimatum game. "
                    "Тебе предлагают часть от 100 ₽. Можешь принять или отказать "
                    "(тогда оба получат 0). Действуй как указанная персона — "
                    "несправедливые офферы отвергай из принципа."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Персона:\n{_persona_block(persona)}\n"
                    f"Предлагают {offer} ₽ из 100 (себе оставляют {100 - offer}). Принять?"
                ),
            },
        ],
        response_model=ResponderMove,
        temperature=0.7,
        max_retries=2,
    )


def run_ultimatum(
    personas: list[Persona],
    offers: list[int] | None = None,
) -> list[UltimatumRound]:
    """Каждая персона: 1 раз proposer + ответы на фиксированные офферы."""
    offers = offers or DEFAULT_OFFERS
    rounds: list[UltimatumRound] = []
    rid = 0

    for persona in personas:
        rid += 1
        prop = propose_as(persona, rid)
        rounds.append(
            UltimatumRound(
                round_id=f"prop_{persona.id}",
                persona_id=persona.id,
                role="proposer",
                offer=prop.share_to_responder,
                accept=None,
                reasoning=prop.reasoning,
            )
        )

        for offer in offers:
            rid += 1
            resp = respond_as(persona, offer)
            rounds.append(
                UltimatumRound(
                    round_id=f"resp_{persona.id}_{offer}",
                    persona_id=persona.id,
                    role="responder",
                    offer=offer,
                    accept=resp.accept,
                    reasoning=resp.reasoning,
                )
            )

    return rounds
