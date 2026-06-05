"""Генератор синтетических заявок на курсы ДПО.

Я не отдаю модели задачу в стиле «придумай 50 любых людей», потому что так
быстро появляются повторяющиеся города, профессии и похожие биографии. Вместо
этого сначала собираю карточки с городом и специальностью, а модель уже
дозаполняет человеческие детали: ФИО, район, курс, стаж и год выпуска.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt

from llm_client import get_model, make_client
from schema import (
    CITIES,
    CITY_DISTRICTS,
    Application,
    get_validation_stats,
    reset_validation_stats,
)


N_APPLICATIONS = 50
CITY_BATCH_SIZE = 5
OUTER_RETRIES = 4
RANDOM_SEED = 20260529

SPECIALITIES = list(Application.model_fields["speciality"].annotation.__args__)
DESIRED_COURSES = list(Application.model_fields["desired_course"].annotation.__args__)

OUT_CSV = Path("applications.csv")
OUT_JSON = Path("applications.json")
OUT_CITIES = Path("cities.png")
OUT_SPECIALITIES = Path("specialities.png")
OUT_CONCLUSIONS = Path("выводы.md")

COURSE_HINTS = {
    "учитель": ("цифровая педагогика", "управление проектами"),
    "врач": ("медицинская информатика", "анализ данных"),
    "инженер": ("бережливое производство", "кибербезопасность"),
    "бухгалтер": ("финансовый менеджмент", "анализ данных"),
    "юрист": ("кибербезопасность", "управление проектами"),
    "HR-специалист": ("HR-аналитика", "управление проектами"),
    "маркетолог": ("анализ данных", "финансовый менеджмент"),
    "менеджер проектов": ("управление проектами", "бережливое производство"),
    "аналитик данных": ("анализ данных", "кибербезопасность"),
    "специалист по охране труда": ("бережливое производство", "управление проектами"),
}

CAREER_NOTES = (
    "хочет перейти в роль методиста и вести внутренние обучения",
    "получил новую зону ответственности после повышения",
    "готовит проект цифровизации в своей организации",
    "ищет программу, которую можно совмещать с плотным графиком",
    "планирует подтвердить компетенции для участия в конкурсе",
    "работает с молодыми сотрудниками и хочет систематизировать опыт",
    "переходит из узкой практики в управленческую роль",
    "выбирает курс под конкретную задачу на работе",
)


@dataclass(frozen=True)
class CohortSlot:
    number: int
    city: str
    speciality: str
    districts: tuple[str, ...]
    suggested_courses: tuple[str, ...]
    career_note: str


SYSTEM_PROMPT = dedent(
    """
    Ты помогаешь учебному центру собрать реалистичные заявки на программы ДПО.
    Это синтетические данные, поэтому не используй известных людей и не повторяй
    шаблонные имена из раза в раз.

    Важные договорённости:
    - ответом должен быть один JSON-объект;
    - город и текущая специальность бери из карточки заявки без изменений;
    - район выбирай только из списка в карточке;
    - курс выбирай из разрешённого списка, лучше из рекомендованных вариантов;
    - возраст, стаж и год выпуска должны выглядеть как одна правдоподобная биография.
    """
).strip()


def build_cohort() -> list[CohortSlot]:
    # Это основная защита от mode collapse: не надеюсь на случайность модели,
    # а заранее раскладываю 50 заявок по городам и специальностям.
    cities = [city for city in CITIES for _ in range(CITY_BATCH_SIZE)]
    specialities = [
        speciality
        for speciality in SPECIALITIES
        for _ in range(N_APPLICATIONS // len(SPECIALITIES))
    ]
    random.shuffle(cities)
    random.shuffle(specialities)

    slots: list[CohortSlot] = []
    for number, (city, speciality) in enumerate(zip(cities, specialities, strict=True), 1):
        slots.append(
            CohortSlot(
                number=number,
                city=city,
                speciality=speciality,
                districts=CITY_DISTRICTS[city],
                suggested_courses=COURSE_HINTS[speciality],
                career_note=random.choice(CAREER_NOTES),
            )
        )
    return slots


def user_card(slot: CohortSlot) -> str:
    return dedent(
        f"""
        Карточка заявки #{slot.number}

        Город: {slot.city}
        Допустимые районы: {", ".join(slot.districts)}
        Текущая специальность: {slot.speciality}
        Все допустимые курсы: {", ".join(DESIRED_COURSES)}
        Рекомендованные курсы для этой истории: {", ".join(slot.suggested_courses)}
        Контекст заявителя: {slot.career_note}

        Сгенерируй одну заявку. Не добавляй лишние поля: мне нужен JSON ровно
        под схему Application.
        """
    ).strip()


def generate_one(slot: CohortSlot) -> tuple[Application, int]:
    client = make_client()
    model = get_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_card(slot)},
    ]

    for attempt in range(1, OUTER_RETRIES + 1):
        # Внутренние ретраи делает make_client(): он повторяет запрос, если JSON
        # не проходит Pydantic-схему. Внешний цикл нужен на случай, если модель
        # формально вернула валидную заявку, но поменяла город или специальность.
        application = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=Application,
            max_retries=3,
            temperature=0.85,
        )
        if application.address.city == slot.city and application.speciality == slot.speciality:
            return application, attempt

        messages.append(
            {
                "role": "user",
                "content": (
                    "Почти хорошо, но карточку нельзя менять. Пересобери JSON так, "
                    f"чтобы address.city был «{slot.city}», а speciality — "
                    f"«{slot.speciality}». Остальные поля оставь правдоподобными."
                ),
            }
        )

    raise RuntimeError(
        f"Не удалось получить заявку для города {slot.city} "
        f"и специальности {slot.speciality}"
    )


def flatten_application(application: Application) -> dict[str, str | int]:
    data = application.model_dump()
    address = data.pop("address")
    # В CSV вложенные объекты неудобны, поэтому address разворачиваю в две
    # отдельные колонки. Так проще строить графики и проверять распределения.
    return {
        "full_name": data["full_name"],
        "age": data["age"],
        "city": address["city"],
        "district": address["district"],
        "speciality": data["speciality"],
        "desired_course": data["desired_course"],
        "years_of_experience": data["years_of_experience"],
        "graduation_year": data["graduation_year"],
    }


def save_csv(applications: list[Application]) -> None:
    rows = [flatten_application(item) for item in applications]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(applications: list[Application]) -> None:
    data = [item.model_dump() for item in applications]
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_bar(counts: Counter[str], title: str, ylabel: str, output: Path) -> None:
    labels, values = zip(*counts.most_common())
    plt.figure(figsize=(11, 5.5))
    colors = plt.cm.Set3(range(len(labels)))
    plt.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.6)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output, dpi=140)
    plt.close()


def top_share(counts: Counter[str]) -> float:
    return max(counts.values()) / sum(counts.values())


def normalized_entropy(counts: Counter[str]) -> float:
    # Энтропия дополняет долю самой частой категории: она показывает, насколько
    # всё распределение похоже на равномерное, а не только кто занял первое место.
    total = sum(counts.values())
    probabilities = [value / total for value in counts.values()]
    entropy = -sum(p * log(p) for p in probabilities if p)
    return entropy / log(len(counts))


def write_conclusions(
    applications: list[Application],
    validator_stats: dict[str, int],
    retry_counts: list[int],
) -> None:
    city_counts = Counter(item.address.city for item in applications)
    speciality_counts = Counter(item.speciality for item in applications)
    top_city, top_city_count = city_counts.most_common(1)[0]
    top_speciality, top_speciality_count = speciality_counts.most_common(1)[0]
    manual_retries = sum(count - 1 for count in retry_counts)
    city_entropy = normalized_entropy(city_counts)
    speciality_entropy = normalized_entropy(speciality_counts)

    validator_text = (
        "В этом прогоне `@field_validator` не поймал реальных ошибок: модель "
        "сразу выдержала список городов и связь возраста с годом выпуска."
        if not validator_stats
        else (
            "В этом прогоне `@field_validator` сработал по-настоящему: "
            + ", ".join(f"{name}={count}" for name, count in sorted(validator_stats.items()))
            + ". Ретраи помогли довести ответы до валидной схемы."
        )
    )

    text = dedent(
        f"""
        # Выводы

        Я не стал полагаться на случайный выбор города: для каждой категории
        заранее выделена квота, а модель заполняет уже «карточку» конкретной
        заявки. Поэтому главный контроль качества — доля самой частой категории:
        если один город или одна специальность резко вырывается вперёд, это
        признак mode collapse. В итоговом наборе топ-город `{top_city}` занимает
        {top_city_count}/50 ({top_city_count / 50:.0%}), топ-специальность
        `{top_speciality}` — {top_speciality_count}/50
        ({top_speciality_count / 50:.0%}). Нормированная энтропия тоже равна
        почти идеальному значению: города — {city_entropy:.2f}, специальности —
        {speciality_entropy:.2f}; это удобно читать как «насколько распределение
        похоже на равномерное».

        {validator_text} Я отдельно считаю и эти ошибки, и ручные повторы из-за
        нарушения карточки заявки: таких повторов было {manual_retries}. Что
        осталось слабым местом: связка «текущая специальность → выбранный курс»
        оценивается промптом и визуальным просмотром, а не строгим валидатором.
        Валидных заявок в итоговом файле: {len(applications)}/50.
        """
    ).strip()

    OUT_CONCLUSIONS.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    random.seed(RANDOM_SEED)
    reset_validation_stats()
    cohort = build_cohort()
    applications: list[Application] = []
    retry_counts: list[int] = []

    for slot in cohort:
        print(f"[{slot.number:02d}/{N_APPLICATIONS}] {slot.city}, {slot.speciality}")
        application, attempts = generate_one(slot)
        applications.append(application)
        retry_counts.append(attempts)
        time.sleep(0.2)

    save_csv(applications)
    save_json(applications)

    city_counts = Counter(item.address.city for item in applications)
    speciality_counts = Counter(item.speciality for item in applications)
    plot_bar(city_counts, "Распределение заявок по городам", "Число заявок", OUT_CITIES)
    plot_bar(
        speciality_counts,
        "Распределение заявок по специальностям",
        "Число заявок",
        OUT_SPECIALITIES,
    )
    write_conclusions(applications, get_validation_stats(), retry_counts)

    print("\nСохранено:")
    for path in (OUT_CSV, OUT_JSON, OUT_CITIES, OUT_SPECIALITIES, OUT_CONCLUSIONS):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
