# Лабораторная 4: RAG по корпусу «Сфера Банк»

Домашнее задание по [семинару 4](https://github.com/paNikitin/gen-ai/tree/main/семинар_4):
сравнение fixed-size и recursive чанкинга на собственном корпусе.

## Корпус

10 внутренних документов продуктовой команды мобильного банка «Сфера Банк»
(исследования, интервью, отчёт поддержки, постмортем, compliance). Тема связана с лаб. 3.

| Параметр | Значение |
|----------|----------|
| Документов | 10 |
| Символов | 31 197 |
| Формат | `.txt` в `data/` |

## Быстрый старт

```bash
pip install -r requirements.txt
cp .env.example .env   # впиши LLM_AUTH_TOKEN

python generate_corpus.py          # пересобрать data/*.txt (опционально)
python eval.py --compare           # ingest + eval обеих стратегий
python pipeline.py ask "Где упоминается СБП?"
```

## Структура

```
Lab4/
├── data/              # корпус + gold.json
├── pipeline.py        # ingest, hybrid RRF, ask
├── eval.py            # hit-rate@5, --compare
├── schema.py          # RAGAnswer
├── generate_corpus.py # генератор корпуса
├── output/            # eval_*.json, ingest_*.json
└── выводы.md          # отчёт
```

## Результаты (кратко)

| Стратегия | Чанков | hit-rate@5 |
|-----------|--------|------------|
| fixed (2000 симв.) | 27 | **0.58** |
| recursive (400/80) | 155 | **0.56** |

Подробный разбор — в `выводы.md`.
