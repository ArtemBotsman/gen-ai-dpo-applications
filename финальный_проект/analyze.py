"""Анализ транскрипта фокус-группы: IE, аспекты, ghost-цитаты, LLM-as-judge."""

from __future__ import annotations

import re

from llm_client import get_model, make_client
from schema import (
    AnalysisJudgeReport,
    FocusGroupTranscript,
    GhostQuoteReport,
    TranscriptAnalysis,
)


def extract_aspects(transcript: FocusGroupTranscript) -> list[TranscriptAnalysis]:
    client = make_client()
    corpus = "\n".join(
        f"{u.speaker_name} ({u.speaker_id}): {u.text}" for u in transcript.utterances
    )
    speakers = sorted({u.speaker_id for u in transcript.utterances})
    results: list[TranscriptAnalysis] = []

    for sid in speakers:
        lines = [u for u in transcript.utterances if u.speaker_id == sid]
        text = "\n".join(f"{u.speaker_name}: {u.text}" for u in lines)
        item = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Извлеки аспекты справедливости из реплик участника фокус-группы. "
                        "Каждый аспект — с дословной цитатой из текста (quote). "
                        "Аспекты: равенство, заслуженность, доверие, риск, альтруизм, "
                        "наказание_за_жадность."
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_model=TranscriptAnalysis,
            temperature=0.0,
            max_retries=2,
        )
        item.speaker_id = sid
        results.append(item)

    return results


def check_ghost_quotes(
    transcript: FocusGroupTranscript,
    analyses: list[TranscriptAnalysis],
) -> GhostQuoteReport:
    """Цитата «ghost», если её нет в исходном транскрипте."""
    corpus = " ".join(u.text.lower() for u in transcript.utterances)
    corpus = re.sub(r"\s+", " ", corpus)
    ghosts: list[str] = []
    total = 0

    for analysis in analyses:
        for asp in analysis.aspects:
            total += 1
            q = asp.quote.strip().lower()
            q_norm = re.sub(r"\s+", " ", q)
            # допускаем подстроку ≥8 символов
            found = False
            if len(q_norm) >= 8:
                found = q_norm in corpus
            else:
                found = any(q_norm in u.text.lower() for u in transcript.utterances)
            if not found:
                ghosts.append(f"{analysis.speaker_id}: «{asp.quote[:80]}»")

    rate = len(ghosts) / total if total else 0.0
    return GhostQuoteReport(
        total_quotes=total,
        ghost_quotes=len(ghosts),
        ghost_rate=round(rate, 3),
        examples=ghosts[:5],
    )


def judge_summary(
    transcript: FocusGroupTranscript,
    ghost: GhostQuoteReport,
) -> AnalysisJudgeReport:
    client = make_client()
    text = "\n".join(f"{u.speaker_name}: {u.text}" for u in transcript.utterances)
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты судья качества анализа фокус-группы. Оцени три утверждения: "
                    "1) участники различаются по взглядам на справедливость; "
                    "2) звучит тема наказания за жадность; "
                    "3) есть консенсус про 50/50. "
                    "supported: yes/partial/no. overall_score 0–1."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Транскрипт:\n{text}\n\n"
                    f"Ghost-цитат: {ghost.ghost_quotes}/{ghost.total_quotes} "
                    f"({ghost.ghost_rate:.0%})"
                ),
            },
        ],
        response_model=AnalysisJudgeReport,
        temperature=0.0,
        max_retries=2,
    )
