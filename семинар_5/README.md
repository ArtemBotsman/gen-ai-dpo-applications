# Лабораторная 5: макро-агент ЦБ/Росстат

Домашнее задание по [семинару 5](https://github.com/paNikitin/gen-ai/tree/main/семинар_5):
ReAct-агент с инструментами, `compare_periods`, лог `trace.jsonl`, eval на 10 вопросов.

## Быстрый старт

```bash
cd starter
pip install -r requirements.txt
cp ../.env.example .env   # впиши LLM_AUTH_TOKEN

python agent.py "Во сколько раз вырос курс USD с января 2022 по апрель 2026?"
python eval.py
```

## Что сделано

- **compare_periods** — 6-й инструмент (сравнение метрик между периодами)
- **trace.jsonl** — JSONL-лог всех шагов (append)
- **eval.py** — 10 вопросов (4 базовых + 6 своих)
- **выводы.md** — отчёт с диагностикой ошибок

## Результаты eval

**10/10** пройдено. Подробности в `starter/eval_results.json` и `выводы.md`.
