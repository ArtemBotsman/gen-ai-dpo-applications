"""Схемы для анализа отзывов из магазина приложений.

Вариант A из ДЗ: вместо участников фокус-группы — Review, вместо concerns — issues.
Аспекты: производительность, дизайн, поддержка, цена, реклама, надёжность.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

IssueCategory = Literal[
    "performance",
    "design",
    "support",
    "price",
    "ads",
    "reliability",
]

AspectType = Literal[
    "performance",
    "design",
    "support",
    "price",
    "ads",
    "reliability",
]

Platform = Literal["ios", "android", "rustore"]


class Issue(BaseModel):
    category: IssueCategory
    severity: int = Field(ge=1, le=5, description="Насколько остро звучит проблема")
    quote: str = Field(min_length=5, description="Дословная цитата из текста отзыва")


class Review(BaseModel):
    author: str = Field(min_length=2, max_length=80)
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    platform: Platform
    review_date: date
    app_version: Optional[str] = Field(default=None, max_length=20)
    body: str = Field(min_length=10, description="Полный текст отзыва")
    issues: list[Issue] = Field(default_factory=list)
    competitor_mentions: list[str] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def review_date_not_in_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("Дата отзыва не может быть в будущем")
        return value


class AspectSentiment(BaseModel):
    aspect: AspectType
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str = Field(min_length=5)
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewAspects(BaseModel):
    author: str
    aspects: list[AspectSentiment]


class ChunkSummary(BaseModel):
    author: str
    key_points: list[str] = Field(min_length=1, max_length=5)
    sentiment: Literal["positive", "negative", "mixed"]


class ReviewsSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0.0, le=1.0)
    summary: str


# Autodiscovery (раунд 2.5)
class DiscoveredAspect(BaseModel):
    name: str = Field(min_length=2, max_length=40)
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str = Field(min_length=5)
    confidence: float = Field(ge=0.0, le=1.0)


class DynamicReviewAspects(BaseModel):
    author: str
    aspects: list[DynamicAspect]


# Multi-doc (раунд 7)
class MultiDocSummary(BaseModel):
    common_themes: list[str] = Field(min_length=1, max_length=8)
    unique_per_source: dict[str, list[str]]
    overall_headline: str
