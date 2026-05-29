"""Схема заявки на ДПО.

Здесь я держу не только типы, но и несколько здравых ограничений: город должен
быть из нашего списка, а возраст, стаж и год выпуска не должны спорить друг с
другом. Так проще отличить аккуратную синтетику от красивого, но странного JSON.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


CITIES = (
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов-на-Дону",
    "Пермь",
)

CITY_DISTRICTS: dict[str, tuple[str, ...]] = {
    "Москва": ("ЦАО", "САО", "ЮЗАО", "ВАО"),
    "Санкт-Петербург": ("Адмиралтейский", "Петроградский", "Приморский", "Московский"),
    "Новосибирск": ("Центральный", "Советский", "Октябрьский", "Ленинский"),
    "Екатеринбург": ("Ленинский", "Кировский", "Чкаловский", "Верх-Исетский"),
    "Казань": ("Вахитовский", "Советский", "Приволжский", "Ново-Савиновский"),
    "Нижний Новгород": ("Нижегородский", "Советский", "Канавинский", "Автозаводский"),
    "Самара": ("Ленинский", "Самарский", "Промышленный", "Октябрьский"),
    "Краснодар": ("Центральный", "Прикубанский", "Западный", "Карасунский"),
    "Ростов-на-Дону": ("Кировский", "Ленинский", "Советский", "Ворошиловский"),
    "Пермь": ("Ленинский", "Свердловский", "Мотовилихинский", "Индустриальный"),
}

Speciality = Literal[
    "учитель",
    "врач",
    "инженер",
    "бухгалтер",
    "юрист",
    "HR-специалист",
    "маркетолог",
    "менеджер проектов",
    "аналитик данных",
    "специалист по охране труда",
]

DesiredCourse = Literal[
    "цифровая педагогика",
    "управление проектами",
    "анализ данных",
    "кибербезопасность",
    "финансовый менеджмент",
    "HR-аналитика",
    "бережливое производство",
    "медицинская информатика",
]

# Счётчик нужен для отчёта: хочется честно сказать, ругался ли валидатор в
# реальном прогоне, а не просто показать, что он написан.
VALIDATOR_REJECTIONS: Counter[str] = Counter()


def reset_validation_stats() -> None:
    VALIDATOR_REJECTIONS.clear()


def get_validation_stats() -> dict[str, int]:
    return dict(VALIDATOR_REJECTIONS)


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=60)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, value: str) -> str:
        if value not in CITIES:
            VALIDATOR_REJECTIONS["city_not_allowed"] += 1
            raise ValueError(f"Город «{value}» не из утверждённого списка")
        return value


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=120)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Speciality
    desired_course: DesiredCourse
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @field_validator("graduation_year")
    @classmethod
    def graduation_year_must_match_age(cls, value: int, info: ValidationInfo) -> int:
        current_year = date.today().year
        age = info.data.get("age")
        if age is None:
            return value

        age_at_graduation = age - (current_year - value)
        if age_at_graduation < 18 or age_at_graduation > 45:
            VALIDATOR_REJECTIONS["graduation_year_age_conflict"] += 1
            raise ValueError(
                "Год выпуска выглядит неправдоподобно: возраст на момент выпуска "
                f"получается {age_at_graduation}"
            )
        return value

    @model_validator(mode="after")
    def experience_must_match_age(self) -> "Application":
        if self.years_of_experience > max(0, self.age - 18):
            VALIDATOR_REJECTIONS["experience_age_conflict"] += 1
            raise ValueError("Стаж не может быть больше профессионального возраста")
        return self
