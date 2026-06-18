# Семинар 6 — Мультиагент: Планировщик-Исполнитель-Критик

Оборачиваем одиночного макро-агента из С5 в паттерн PWC: Планировщик декомпозирует
вопрос, Исполнитель отвечает на подвопросы, Критик проверяет результат.

## Домашнее задание

- `validate_plan()` — ловит выдуманные tools и битый `depends_on`
- параллельное исполнение уровней (`_topological_levels` + `ThreadPoolExecutor`)
- замер угодливости Критика (`critic_sycophancy.py`)
- eval 6 вопросов × 3 конфигурации (`eval_pwc.py`)

```bash
cd starter
pip install -r requirements.txt
cp .env.example .env   # токен LLM

python orchestrator.py "Какова накопленная инфляция с января 2022 по март 2026?"
python orchestrator.py --benchmark "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"
python critic_sycophancy.py
python eval_pwc.py -n 3
```

Отчёт: `выводы.md`
