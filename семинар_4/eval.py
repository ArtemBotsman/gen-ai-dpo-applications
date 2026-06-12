"""
Eval по gold.json. Метрика: hit-rate@K на уровне документа-источника.

Команды:
    python eval.py --chunking fixed
    python eval.py --chunking recursive
    python eval.py --compare          # обе стратегии + сводка
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import _save_active_chunking, get_collection, hybrid_retrieve, ingest

GOLD_PATH = Path(__file__).parent / "data" / "gold.json"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def run_eval(chunking: str, k: int = 5, verbose: bool = True, do_ingest: bool = False) -> dict:
    if do_ingest:
        print(f"\n>>> ingest --chunking {chunking}")
        ingest(chunking)

    _save_active_chunking(chunking)
    col = get_collection(chunking)
    if col.count() == 0:
        raise RuntimeError(f"Коллекция {chunking} пуста. Запусти: python pipeline.py ingest --chunking {chunking}")

    gold = load_gold()
    total = 0.0
    results = []

    label = "FIXED (2000 символов)" if chunking == "fixed" else "RECURSIVE (400/80)"
    if verbose:
        print(f"\n=== {label} ===\n")

    for item in gold:
        hits = hybrid_retrieve(item["question"], k=k, chunking=chunking)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]
        score = hit_rate(retrieved_ids, item["gold_sources"])
        total += score

        row = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "score": score,
            "gold": item["gold_sources"],
            "retrieved_sources": retrieved_sources,
            "retrieved_ids": retrieved_ids,
            "note": item.get("note", ""),
        }
        results.append(row)

        if verbose:
            mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
            print(f"  [{item['id']:2d}] {item['type']:25s}  hit@{k} = {score:.2f}  {mark}  {item['question'][:60]}")

    mean = total / len(gold)
    if verbose:
        print(f"\n  ИТОГО: hit-rate@{k} = {mean:.2f}  ({total:.1f} / {len(gold)})")

    payload = {"chunking": chunking, "k": k, "mean": mean, "total_hits": total, "n_questions": len(gold), "results": results}
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"eval_{chunking}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def compare(k: int = 5, do_ingest: bool = True) -> dict:
    fixed = run_eval("fixed", k=k, verbose=True, do_ingest=do_ingest)
    recursive = run_eval("recursive", k=k, verbose=True, do_ingest=do_ingest)

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА")
    print("=" * 60)
    print(f"  fixed     hit-rate@{k} = {fixed['mean']:.2f}")
    print(f"  recursive hit-rate@{k} = {recursive['mean']:.2f}")
    winner = "recursive" if recursive["mean"] > fixed["mean"] else ("fixed" if fixed["mean"] > recursive["mean"] else "tie")
    print(f"  победитель: {winner}")

    # вопросы, где стратегии расходятся
    print("\n  Расхождения (одна стратегия = 1.0, другая < 1.0):")
    for f_row, r_row in zip(fixed["results"], recursive["results"]):
        if f_row["score"] != r_row["score"]:
            print(f"    Q{f_row['id']}: fixed={f_row['score']:.2f}, recursive={r_row['score']:.2f} — {f_row['type']}")

    summary = {
        "k": k,
        "fixed_mean": fixed["mean"],
        "recursive_mean": recursive["mean"],
        "winner": winner,
        "fixed": fixed,
        "recursive": recursive,
    }
    (OUTPUT_DIR / "eval_compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunking", choices=["fixed", "recursive"], default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-ingest", action="store_true", help="не переиндексировать перед eval")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare(k=args.k, do_ingest=not args.no_ingest)
    elif args.chunking:
        run_eval(args.chunking, k=args.k, verbose=not args.quiet, do_ingest=not args.no_ingest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
