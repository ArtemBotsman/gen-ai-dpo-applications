"""Pydantic-схемы для проекта «Лаборатория справедливости»."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Education = Literal["среднее", "среднее специальное", "высшее", "неоконченное высшее"]
IncomeBand = Literal["низкий", "средний", "высокий"]
CitySize = Literal["мегаполис", "город", "малый город"]


class DemProfile(BaseModel):
    """Якорь из реального опросника (ESS-inspired)."""

    id: str
    age: int = Field(ge=18, le=75)
    gender: Literal["мужской", "женский"]
    education: Education
    income_band: IncomeBand
    city_size: CitySize
    region: str
    ess_trust_baseline: float = Field(
        ge=0.0, le=10.0, description="Средний trust в ESS для похожей когорты"
    )


class Persona(BaseModel):
    """Синтетическая персона для silicon sampling."""

    id: str
    name: str = Field(min_length=3, max_length=60)
    age: int = Field(ge=18, le=75)
    gender: Literal["мужской", "женский"]
    education: Education
    income_band: IncomeBand
    city_size: CitySize
    region: str
    bio: str = Field(min_length=20, max_length=400)
    fairness_note: str = Field(
        min_length=10,
        max_length=200,
        description="Как персона понимает справедливость",
    )

    @field_validator("age")
    @classmethod
    def age_plausible(cls, v: int) -> int:
        if v < 18 or v > 75:
            raise ValueError("возраст персоны вне диапазона 18–75")
        return v


class FocusUtterance(BaseModel):
    speaker_id: str
    speaker_name: str
    text: str = Field(min_length=5, max_length=500)


class FocusGroupTranscript(BaseModel):
    topic: str
    utterances: list[FocusUtterance] = Field(min_length=6, max_length=40)
    generated_at: date


class FairnessAspect(BaseModel):
    aspect: Literal[
        "равенство",
        "заслуженность",
        "доверие",
        "риск",
        "альтруизм",
        "наказание_за_жадность",
    ]
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str = Field(min_length=5)
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptAnalysis(BaseModel):
    speaker_id: str
    aspects: list[FairnessAspect] = Field(min_length=1, max_length=8)


class SurveyAnswer(BaseModel):
    persona_id: str
    trust_people: int = Field(ge=1, le=5, description="Людям вообще можно доверять?")
    trust_strangers: int = Field(ge=1, le=5)
    risk_appetite: int = Field(ge=1, le=5, description="Готовность к риску")
    fairness_priority: int = Field(
        ge=1, le=5, description="Насколько важна справедливость vs выгода"
    )
    min_acceptable_share: int = Field(
        ge=0, le=50, description="Минимальная доля второму в ultimatum, ₽ из 100"
    )
    would_punish_unfair: int = Field(
        ge=1, le=5, description="Отказать себе, чтобы наказать несправедливость"
    )

    @field_validator("min_acceptable_share")
    @classmethod
    def share_not_above_half(cls, v: int) -> int:
        if v > 50:
            raise ValueError("минимально приемлемая доля второму не может быть >50")
        return v


class ProposerMove(BaseModel):
    share_to_responder: int = Field(ge=0, le=100)
    reasoning: str = Field(default="", max_length=200)


class ResponderMove(BaseModel):
    accept: bool
    reasoning: str = Field(default="", max_length=200)


class UltimatumRound(BaseModel):
    round_id: str
    persona_id: str
    role: Literal["proposer", "responder"]
    offer: int = Field(ge=0, le=100)
    accept: bool | None = None
    reasoning: str = ""


class GhostQuoteReport(BaseModel):
    total_quotes: int
    ghost_quotes: int
    ghost_rate: float
    examples: list[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    claim: str
    supported: Literal["yes", "partial", "no"]
    evidence: str
    comment: str


class AnalysisJudgeReport(BaseModel):
    verdicts: list[JudgeVerdict]
    overall_score: float = Field(ge=0.0, le=1.0)
    summary: str
