"""Демо-данные для прогона без LLM (python pipeline.py --mock)."""

from __future__ import annotations

from datetime import date

from schema import (
    AnalysisJudgeReport,
    FairnessAspect,
    FocusGroupTranscript,
    FocusUtterance,
    GhostQuoteReport,
    JudgeVerdict,
    Persona,
    SurveyAnswer,
    TranscriptAnalysis,
    UltimatumRound,
)


def mock_personas() -> list[Persona]:
    return [
        Persona(
            id="p1", name="Алина К.", age=24, gender="женский",
            education="неоконченное высшее", income_band="низкий",
            city_size="мегаполис", region="Москва",
            bio="Студентка, подрабатывает репетитором. Живёт с родителями.",
            fairness_note="Считает, что делить нужно поровну — иначе это жадность.",
        ),
        Persona(
            id="p2", name="Руслан М.", age=38, gender="мужской",
            education="высшее", income_band="средний",
            city_size="город", region="Казань",
            bio="Инженер на заводе, двое детей.",
            fairness_note="Готов уступить, но не больше четверти — семья важнее.",
        ),
        Persona(
            id="p3", name="Нина П.", age=52, gender="женский",
            education="среднее специальное", income_band="средний",
            city_size="малый город", region="Вологодская обл.",
            bio="Медсестра в поликлинике.",
            fairness_note="Несправедливо, когда сильный забирает всё — надо помогать слабому.",
        ),
        Persona(
            id="p4", name="Дмитрий В.", age=31, gender="мужской",
            education="высшее", income_band="высокий",
            city_size="мегаполис", region="Санкт-Петербург",
            bio="Менеджер в IT, снимает квартиру.",
            fairness_note="Рационально предложить 30–40, чтобы точно приняли.",
        ),
        Persona(
            id="p5", name="Гульнара С.", age=45, gender="женский",
            education="среднее", income_band="низкий",
            city_size="малый город", region="Башкортостан",
            bio="Продавец в магазине.",
            fairness_note="Если предложат копейки — лучше отказать, пусть тоже ноль.",
        ),
        Persona(
            id="p6", name="Игорь Л.", age=29, gender="мужской",
            education="высшее", income_band="средний",
            city_size="город", region="Новосибирск",
            bio="Бухгалтер в строительной фирме.",
            fairness_note="50 на 50 — золотая середина, так учили в универе.",
        ),
    ]


def mock_transcript(personas: list[Persona]) -> FocusGroupTranscript:
    lines = [
        ("p1", "Я за fifty-fifty. Иначе второй просто откажется."),
        ("p2", "Поровну — это идеал, но в жизни оставлю себе больше."),
        ("p3", "Жадность надо наказывать — откажусь, если мало."),
        ("p4", "30–40 рублей второму — разумный компромисс."),
        ("p5", "10 рублей — это оскорбление, не возьму."),
        ("p6", "Зависит от ситуации, но ниже 25 не приму."),
    ]
    utterances = []
    for pid, text in lines:
        p = next(x for x in personas if x.id == pid)
        utterances.append(FocusUtterance(speaker_id=pid, speaker_name=p.name, text=text))
    return FocusGroupTranscript(
        topic="Как честно разделить 100 ₽?",
        utterances=utterances,
        generated_at=date.today(),
    )


def mock_survey() -> list[SurveyAnswer]:
    return [
        SurveyAnswer(persona_id="p1", trust_people=2, trust_strangers=2, risk_appetite=3,
                     fairness_priority=5, min_acceptable_share=45, would_punish_unfair=4),
        SurveyAnswer(persona_id="p2", trust_people=3, trust_strangers=3, risk_appetite=3,
                     fairness_priority=4, min_acceptable_share=30, would_punish_unfair=3),
        SurveyAnswer(persona_id="p3", trust_people=4, trust_strangers=3, risk_appetite=2,
                     fairness_priority=5, min_acceptable_share=40, would_punish_unfair=5),
        SurveyAnswer(persona_id="p4", trust_people=3, trust_strangers=2, risk_appetite=4,
                     fairness_priority=3, min_acceptable_share=25, would_punish_unfair=2),
        SurveyAnswer(persona_id="p5", trust_people=4, trust_strangers=3, risk_appetite=2,
                     fairness_priority=5, min_acceptable_share=35, would_punish_unfair=5),
        SurveyAnswer(persona_id="p6", trust_people=3, trust_strangers=3, risk_appetite=3,
                     fairness_priority=4, min_acceptable_share=40, would_punish_unfair=3),
    ]


def mock_ultimatum() -> list[UltimatumRound]:
    rounds = []
    proposer_offers = {"p1": 50, "p2": 35, "p3": 45, "p4": 38, "p5": 42, "p6": 50}
    responder_accepts = {
        10: {"p1": False, "p2": False, "p3": False, "p4": False, "p5": False, "p6": False},
        20: {"p1": False, "p2": True, "p3": False, "p4": True, "p5": False, "p6": True},
        30: {"p1": True, "p2": True, "p3": True, "p4": True, "p5": False, "p6": True},
        40: {"p1": True, "p2": True, "p3": True, "p4": True, "p5": True, "p6": True},
        50: {"p1": True, "p2": True, "p3": True, "p4": True, "p5": True, "p6": True},
    }
    for pid, offer in proposer_offers.items():
        rounds.append(UltimatumRound(
            round_id=f"prop_{pid}", persona_id=pid, role="proposer",
            offer=offer, accept=None, reasoning="демо",
        ))
    for offer, by_persona in responder_accepts.items():
        for pid, acc in by_persona.items():
            rounds.append(UltimatumRound(
                round_id=f"resp_{pid}_{offer}", persona_id=pid, role="responder",
                offer=offer, accept=acc, reasoning="демо",
            ))
    return rounds


def mock_aspects() -> list[TranscriptAnalysis]:
    return [
        TranscriptAnalysis(
            speaker_id="p1",
            aspects=[FairnessAspect(
                aspect="равенство", sentiment="positive",
                quote="Я за fifty-fifty", confidence=0.9,
            )],
        ),
        TranscriptAnalysis(
            speaker_id="p5",
            aspects=[FairnessAspect(
                aspect="наказание_за_жадность", sentiment="negative",
                quote="10 рублей — это оскорбление", confidence=0.85,
            )],
        ),
    ]


def mock_ghost() -> GhostQuoteReport:
    return GhostQuoteReport(total_quotes=2, ghost_quotes=0, ghost_rate=0.0, examples=[])


def mock_judge() -> AnalysisJudgeReport:
    return AnalysisJudgeReport(
        verdicts=[
            JudgeVerdict(claim="участники различаются", supported="yes",
                         evidence="разные пороги min share", comment=""),
            JudgeVerdict(claim="наказание за жадность", supported="yes",
                         evidence="p5, p3", comment=""),
            JudgeVerdict(claim="консенсус 50/50", supported="partial",
                         evidence="p1, p6", comment=""),
        ],
        overall_score=0.78,
        summary="Есть расхождение по порогу справедливости.",
    )
