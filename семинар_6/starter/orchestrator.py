"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

Домашнее задание С6:
- validate_plan между Планировщиком и Исполнителем
- параллельное исполнение уровней (_topological_levels + execute_level)
- replan / rework ветки
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана (пустой — всё ок)."""
    errors: list[str] = []
    ids = {sq.id for sq in plan.subquestions}
    for sq in plan.subquestions:
        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(f"подвопрос {sq.id}: неизвестный инструмент '{tool}'")
        for dep in sq.depends_on:
            if dep not in ids:
                errors.append(f"подвопрос {sq.id}: depends_on ссылается на id {dep}, которого нет")
            if dep == sq.id:
                errors.append(f"подвопрос {sq.id}: зависит сам от себя")
    return errors


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Плоский список (для обратной совместимости)."""
    return [sq for level in _topological_levels(subqs) for sq in level]


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Список уровней: внутри уровня зависимостей нет, между уровнями — есть."""
    by_id = {s.id: s for s in subqs}
    depth_cache: dict[int, int] = {}

    def depth(node_id: int, path: list[int]) -> int:
        if node_id in path:
            raise ValueError(f"Цикл в depends_on: {path + [node_id]}")
        if node_id not in by_id:
            return 0
        if node_id in depth_cache:
            return depth_cache[node_id]
        deps = [d for d in by_id[node_id].depends_on if d in by_id]
        if not deps:
            depth_cache[node_id] = 0
        else:
            depth_cache[node_id] = max(depth(d, path + [node_id]) for d in deps) + 1
        return depth_cache[node_id]

    for sq in subqs:
        depth(sq.id, [])

    if not depth_cache:
        return []

    max_d = max(depth_cache.values())
    levels: list[list[SubQuestion]] = [[] for _ in range(max_d + 1)]
    for sq in subqs:
        levels[depth_cache[sq.id]].append(sq)
    return levels


def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
    *,
    parallel: bool = True,
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня (параллельно, если их несколько)."""
    if not level:
        return {}

    if parallel and len(level) > 1:
        out: dict[int, WorkerAnswer] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(level))) as ex:
            futures = {ex.submit(worker, sq, prev_answers): sq for sq in level}
            for fut in futures:
                sq = futures[fut]
                out[sq.id] = fut.result()
        return out

    return {sq.id: worker(sq, prev_answers) for sq in level}


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    client = make_raw_client()
    model = get_model()
    facts = "\n".join(
        f"  {i}. {answers[i].answer}" for i in sorted(answers) if i in answers
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Собери из фактов короткий ответ пользователю (1-2 фразы). "
                    "Только числа из фактов, без выдумки."
                ),
            },
            {
                "role": "user",
                "content": f"Вопрос: {question}\n\nФакты Исполнителей:\n{facts or '(пусто)'}",
            },
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip() or " · ".join(
        a.answer for a in answers.values()
    )


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    parallel: bool = True,
    plan_retries: int = 3,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []
    feedback: str | None = None
    plan: Plan | None = None

    for attempt in range(plan_retries):
        plan = planner(question, feedback=feedback)
        trace.append(
            {
                "iter": 0,
                "kind": "plan",
                "attempt": attempt,
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            }
        )
        if verbose:
            print(f"\n[plan] {plan.reasoning}")
            for sq in plan.subquestions:
                print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

        if use_validator:
            errors = validate_plan(plan)
            if errors:
                trace.append({"iter": 0, "kind": "validate_fail", "errors": errors})
                if verbose:
                    print(f"  [validator] ❌ {errors}")
                feedback = f"Инструменты не существуют или план некорректен: {errors}"
                continue
            if verbose:
                print("  [validator] ✅ план ок")
        break
    else:
        return {
            "answer": None,
            "error": "не удалось получить валидный план",
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
        }

    assert plan is not None
    answers: dict[int, WorkerAnswer] = {}

    for iter_num in range(1, max_iter + 1):
        answers = {}
        levels = _topological_levels(plan.subquestions)
        for level in levels:
            batch = execute_level(level, answers, parallel=parallel)
            answers.update(batch)
            for sq in level:
                ans = answers[sq.id]
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        if verdict.action == "replan":
            feedback = verdict.reason
            plan = planner(question, feedback=feedback)
            if use_validator:
                errors = validate_plan(plan)
                if errors:
                    feedback = f"Инструменты не существуют: {errors}. {verdict.reason}"
                    plan = planner(question, feedback=feedback)
            trace.append(
                {
                    "iter": iter_num,
                    "kind": "replan",
                    "reasoning": plan.reasoning,
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                }
            )
            continue

        if verdict.action == "rework":
            feedback = (
                f"Переделать подвопросы {verdict.rework_ids}: {verdict.reason}"
            )
            plan = planner(question, feedback=feedback)
            if use_validator:
                errors = validate_plan(plan)
                while errors:
                    feedback = f"Инструменты не существуют: {errors}. {verdict.reason}"
                    plan = planner(question, feedback=feedback)
                    errors = validate_plan(plan)
            continue

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def benchmark_parallel(question: str, *, runs: int = 1) -> dict[str, float]:
    """Замер времени последовательного vs параллельного исполнения."""
    seq_times: list[float] = []
    par_times: list[float] = []

    for _ in range(runs):
        t0 = time.perf_counter()
        run_pwc(question, max_iter=2, verbose=False, use_validator=True, parallel=False)
        seq_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        run_pwc(question, max_iter=2, verbose=False, use_validator=True, parallel=True)
        par_times.append(time.perf_counter() - t0)

    seq_avg = sum(seq_times) / len(seq_times)
    par_avg = sum(par_times) / len(par_times)
    return {
        "sequential_sec": round(seq_avg, 2),
        "parallel_sec": round(par_avg, 2),
        "speedup": round(seq_avg / par_avg, 2) if par_avg > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-validator", action="store_true")
    ap.add_argument("--sequential", action="store_true", help="без параллельных workers")
    ap.add_argument("--benchmark", action="store_true", help="замер seq vs parallel")
    ap.add_argument("--trace", type=Path, default=None)
    args = ap.parse_args()

    q = " ".join(args.query)

    if args.benchmark:
        stats = benchmark_parallel(q)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    res = run_pwc(
        q,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        use_validator=not args.no_validator,
        parallel=not args.sequential,
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps({"query": q, **_serialize(res)}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
