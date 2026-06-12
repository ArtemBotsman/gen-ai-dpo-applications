# Лабораторная 3: пайплайн анализа отзывов (отлично)

Домашнее задание по [семинару 3](https://github.com/paNikitin/gen-ai/tree/main/семинар_3):
свой текстовый пайплайн на отзывах из магазина приложений (вариант A).

## Реализованные техники

**Обязательные (4):**
1. IE — `extract_reviews()` → `reviews.json`
2. Аспектный анализ — фиксированный `Literal` → `aspects.json` + `heatmap.png`
3. Map-Reduce — параллельный MAP по отзывам + REDUCE → `summary.json`
4. LLM-as-judge → `judge_report.json`

**Для «отлично» (2 из опций):**
- **Autodiscovery (2.5)** — `discovered_aspects.json`, `aspects_discovered.json`, `aspect_comparison.json`
- **Multi-doc (7)** — 5 источников в `input/`, `multi_doc_aspects.csv`, `cross_source_pivot.csv`, `multi_doc_summary.json`

## Структура

```
Lab3/
├── schema.py
├── prompts.py
├── pipeline.py          # analyze(input_path)
├── generate_input.py    # 40 отзывов + 5 источников
├── input/
│   ├── reviews.txt      # полный датасет
│   ├── ios_appstore.txt
│   ├── android_play.txt
│   ├── rustore.txt
│   ├── wave_summer_2025.txt
│   └── wave_autumn_2025.txt
├── output/              # артефакты прогона
└── выводы.md
```

## Запуск

```bash
cd Lab3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env по образцу .env.example

python generate_input.py
python pipeline.py input output
```

Полный прогон занимает ~7–8 минут (multi-doc — самая тяжёлая часть).

## Артефакты output/

| Файл | Что это |
|---|---|
| `reviews.json` | IE |
| `aspects.json`, `heatmap.png` | фиксированные аспекты |
| `discovered_aspects.json` | autodiscovery, стадия A |
| `aspects_discovered.json` | классификация по найденным темам |
| `aspect_comparison.json` | сравнение Literal vs discovered |
| `summary.json` | Map-Reduce |
| `judge_report.json` | оценка судьи |
| `multi_doc_*.csv/json` | multi-doc сводка |
| `metrics.json` | метрики прогона |
