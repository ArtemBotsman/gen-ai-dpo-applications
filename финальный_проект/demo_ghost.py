"""
Демо ghost-цитаты: подкладываем выдуманную фразу и прогоняем проверку.

Запуск:
    python demo_ghost.py
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from analyze import check_ghost_quotes
from schema import FairnessAspect, FocusGroupTranscript, TranscriptAnalysis

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

# Выдуманная цитата — звучит правдоподобно, но её НЕТ в focus_transcript.json
INVENTED_GHOST = {
    "speaker_id": "p2",
    "speaker_name": "Алексей Иванов",
    "aspect": "наказание_за_жадность",
    "sentiment": "negative",
    "quote": (
        "Если кто-то предложит мне 5 рублей из ста — я сожгу эти деньги, "
        "лишь бы жадный не получил ни копейки. Так мне отец в девяностые учил."
    ),
    "why_invented": (
        "У p2 (Алексей Иванов) в транскрипте только реплики про «поровну — честно» "
        "и добровольную помощь нуждающимся. Про «5 рублей», «сожгу» и «отец в девяностые» "
        "он никогда не говорил — типичная галлюцинация аналитика."
    ),
}


def main():
    transcript = FocusGroupTranscript.model_validate(
        json.loads((OUT / "focus_transcript.json").read_text(encoding="utf-8"))
    )
    aspects_raw = json.loads((OUT / "transcript_aspects.json").read_text(encoding="utf-8"))
    analyses = [TranscriptAnalysis.model_validate(x) for x in aspects_raw]

    # Чистый прогон (как было)
    clean = check_ghost_quotes(transcript, analyses)

    # Подкладываем призрака в p2
    poisoned = deepcopy(analyses)
    p2 = next(a for a in poisoned if a.speaker_id == "p2")
    p2.aspects.append(
        FairnessAspect(
            aspect=INVENTED_GHOST["aspect"],
            sentiment=INVENTED_GHOST["sentiment"],
            quote=INVENTED_GHOST["quote"],
            confidence=0.88,
        )
    )

    ghost = check_ghost_quotes(transcript, poisoned)

    # Сохраняем всё наглядно
    example = {
        "invented_quote": INVENTED_GHOST,
        "check_before_injection": clean.model_dump(),
        "check_after_injection": ghost.model_dump(),
        "how_detection_works": (
            "Программа собирает весь текст фокус-группы в одну строку и ищет "
            "каждую цитату из анализа как подстроку. Если фразы нет — это ghost."
        ),
    }
    (OUT / "ghost_quote_example.json").write_text(
        json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    poisoned_path = OUT / "transcript_aspects_with_ghost.json"
    poisoned_path.write_text(
        json.dumps([a.model_dump() for a in poisoned], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT / "ghost_quotes_after_demo.json").write_text(
        json.dumps(ghost.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== Демо ghost-цитаты ===\n")
    print(f"ДО подложки:  {clean.ghost_quotes}/{clean.total_quotes} ghost "
          f"({clean.ghost_rate:.0%})")
    print(f"ПОСЛЕ подложки: {ghost.ghost_quotes}/{ghost.total_quotes} ghost "
          f"({ghost.ghost_rate:.0%})\n")
    print("Выдуманная цитата (её нет в транскрипте):")
    print(f"  [{INVENTED_GHOST['speaker_id']}] «{INVENTED_GHOST['quote'][:90]}…»\n")
    print("Проверка засчитала:")
    for ex in ghost.examples:
        print(f"  👻 {ex}")
    print(f"\nСохранено:")
    print(f"  {OUT / 'ghost_quote_example.json'}")
    print(f"  {poisoned_path}")
    print(f"  {OUT / 'ghost_quotes_after_demo.json'}")


if __name__ == "__main__":
    main()
