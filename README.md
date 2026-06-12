# Практическое применение генеративного ИИ — домашние работы

Материалы курса на базе репозитория [paNikitin/gen-ai](https://github.com/paNikitin/gen-ai).

## Семинар 2 — синтетические заявки на ДПО

Папка: `семинар_2/starter`

Генерация 50 заявок на курсы повышения квалификации через `make_client()`, Pydantic-схему и стратификацию по городам и специальностям.

```bash
cd семинар_2/starter
pip install -r requirements.txt
python generator.py
```

Результат: `50/50` валидных заявок, максимум по городу и специальности — по `10%`.

## Семинар 3 — пайплайн анализа отзывов

Папка: `семинар_3`

Анализ отзывов на мобильное приложение: IE → аспекты → autodiscovery → Map-Reduce → judge → multi-doc.

```bash
cd семинар_3
pip install -r requirements.txt
python generate_input.py
python pipeline.py input output
```

## Настройка LLM

Токен кладём в локальный `.env` (см. `.env.example` в каждой папке). В репозиторий `.env` не коммитим.

```env
LLM_BASE_URL=https://api.deepseek.com
LLM_AUTH_TOKEN=твой_токен
LLM_MODEL=deepseek-v4-flash
```
