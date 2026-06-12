"""Пайплайн анализа отзывов: IE → аспекты → autodiscovery → Map-Reduce → judge → multi-doc.

Запуск:
    python pipeline.py input
    python pipeline.py input/reviews.txt output
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import ValidationError

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    DISCOVER_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    MULTI_DOC_SYSTEM,
    REDUCE_SYSTEM,
    REDUCE_SYSTEM_STRICT,
)
from schema import (
    ChunkSummary,
    DiscoveredAspects,
    DynamicReviewAspects,
    JudgeReport,
    MultiDocSummary,
    Review,
    ReviewAspects,
    ReviewsSummary,
)

ALL_ASPECTS = [
    "performance",
    "design",
    "support",
    "price",
    "ads",
    "reliability",
]

MULTI_DOC_SOURCES = [
    "ios_appstore",
    "android_play",
    "rustore",
    "wave_summer_2025",
    "wave_autumn_2025",
]

INPUT_COST_PER_1M = 0.27
OUTPUT_COST_PER_1M = 1.10

client = make_client()
MODEL = get_model()


@dataclass
class PipelineStats:
    started_at: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    validation_errors: int = 0
    ghost_quotes: list[tuple[str, str]] = field(default_factory=list)
    ghost_quotes_discovered: list[tuple[str, str]] = field(default_factory=list)
    judge_retries: int = 0

    def add_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * INPUT_COST_PER_1M
            + self.completion_tokens / 1_000_000 * OUTPUT_COST_PER_1M
        )


def call_model(response_model, messages: list[dict], *, temperature: float = 0.0, stats: PipelineStats):
    result, response = client.chat.completions.create(
        model=MODEL,
        response_model=response_model,
        max_retries=3,
        temperature=temperature,
        messages=messages,
        with_completion=True,
    )
    stats.add_usage(response)
    return result


def load_corpus(input_path: Path) -> str:
    if input_path.is_dir():
        combined = input_path / "reviews.txt"
        if combined.exists():
            return combined.read_text(encoding="utf-8")
        parts = sorted(input_path.glob("*.txt"))
        if not parts:
            raise FileNotFoundError(f"В {input_path} нет .txt файлов")
        return "\n\n".join(p.read_text(encoding="utf-8") for p in parts)
    return input_path.read_text(encoding="utf-8")


def list_multi_doc_sources(input_dir: Path) -> list[Path]:
    paths = [input_dir / f"{name}.txt" for name in MULTI_DOC_SOURCES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Для multi-doc не хватает файлов: {missing}. Запусти: python generate_input.py"
        )
    return paths


def split_by_review(corpus: str) -> list[str]:
    header_sep = corpus.find("===")
    body = corpus[header_sep:] if header_sep != -1 else corpus
    chunks = [chunk.strip() for chunk in re.split(r"\n---\n", body) if chunk.strip()]
    return [chunk for chunk in chunks if "Автор:" in chunk]


def extract_reviews(corpus: str, stats: PipelineStats) -> list[Review]:
    try:
        return call_model(
            list[Review],
            [
                {"role": "system", "content": IE_SYSTEM},
                {"role": "user", "content": corpus},
            ],
            stats=stats,
        )
    except ValidationError:
        stats.validation_errors += 1
        raise


def extract_aspects(corpus: str, stats: PipelineStats) -> list[ReviewAspects]:
    return call_model(
        list[ReviewAspects],
        [
            {"role": "system", "content": ASPECTS_SYSTEM},
            {"role": "user", "content": corpus},
        ],
        stats=stats,
    )


def discover_aspects(corpus: str, stats: PipelineStats) -> DiscoveredAspects:
    return call_model(
        DiscoveredAspects,
        [
            {"role": "system", "content": DISCOVER_SYSTEM},
            {"role": "user", "content": corpus},
        ],
        stats=stats,
    )


def extract_with_discovered(
    corpus: str,
    discovered: DiscoveredAspects,
    stats: PipelineStats,
) -> list[DynamicReviewAspects]:
    dynamic_block = "\n".join(
        f"- {item.name}: {item.description}" for item in discovered.aspects
    )
    system_prompt = (
        ASPECTS_SYSTEM
        + "\n\nИспользуй СТРОГО эти обнаруженные аспекты:\n"
        + dynamic_block
    )
    return call_model(
        list[DynamicReviewAspects],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": corpus},
        ],
        stats=stats,
    )


def compare_aspect_sets(
    fixed: list[ReviewAspects],
    discovered: list[DynamicReviewAspects],
    discovered_schema: DiscoveredAspects,
) -> dict:
    fixed_used = {aspect.aspect for review in fixed for aspect in review.aspects}
    dynamic_used = {aspect.aspect for review in discovered for aspect in review.aspects}
    discovered_names = {item.name for item in discovered_schema.aspects}
    literal_set = set(ALL_ASPECTS)

    return {
        "fixed_literal_aspects": sorted(literal_set),
        "fixed_used_in_run": sorted(fixed_used),
        "discovered_topic_names": sorted(discovered_names),
        "dynamic_used_in_run": sorted(dynamic_used),
        "only_in_discovered_run": sorted(dynamic_used - fixed_used),
        "only_in_fixed_run": sorted(fixed_used - dynamic_used),
        "invented_not_in_literal": sorted(dynamic_used - literal_set),
        "literal_never_mentioned": sorted(literal_set - fixed_used),
    }


def check_quotes_generic(
    aspects: list,
    corpus: str,
    author_attr: str = "author",
    quote_attr: str = "quote",
    aspects_attr: str = "aspects",
) -> list[tuple[str, str]]:
    text = corpus.lower()
    ghosts: list[tuple[str, str]] = []
    for review in aspects:
        author = getattr(review, author_attr)
        for aspect in getattr(review, aspects_attr):
            quote = getattr(aspect, quote_attr)
            probe = quote.strip().lower()[:30]
            if probe and probe not in text:
                ghosts.append((author, quote))
    return ghosts


def build_heatmap(aspects: list[ReviewAspects], out_path: Path) -> None:
    authors = [item.author for item in aspects]
    sent_to_num = {"positive": 1, "negative": -1, "neutral": 0}
    matrix = np.full((len(authors), len(ALL_ASPECTS)), np.nan)

    for row, review in enumerate(aspects):
        for aspect in review.aspects:
            if aspect.aspect in ALL_ASPECTS:
                col = ALL_ASPECTS.index(aspect.aspect)
                matrix[row, col] = sent_to_num[aspect.sentiment]

    plt.figure(figsize=(10, max(4, len(authors) * 0.35)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        xticklabels=ALL_ASPECTS,
        yticklabels=authors,
        center=0,
        cmap="RdYlGn",
        cbar_kws={"label": "sentiment"},
    )
    plt.title("Тональность по аспектам (отзыв × аспект)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_chunk(chunk: str, stats: PipelineStats) -> ChunkSummary:
    return call_model(
        ChunkSummary,
        [
            {"role": "system", "content": CHUNK_SYSTEM},
            {"role": "user", "content": chunk},
        ],
        stats=stats,
    )


def reduce_summaries(
    summaries: list[ChunkSummary],
    stats: PipelineStats,
    *,
    strict: bool = False,
) -> ReviewsSummary:
    joined = "\n\n".join(
        f"## {item.author} ({item.sentiment})\n"
        + "\n".join(f"- {point}" for point in item.key_points)
        for item in summaries
    )
    system_prompt = REDUCE_SYSTEM_STRICT if strict else REDUCE_SYSTEM
    return call_model(
        ReviewsSummary,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": joined},
        ],
        stats=stats,
    )


def summarize_reviews(corpus: str, stats: PipelineStats, *, strict: bool = False) -> ReviewsSummary:
    chunks = split_by_review(corpus)
    print(f"  [MR] MAP: {len(chunks)} отзывов...")
    summaries: list[ChunkSummary | None] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(summarize_chunk, chunk, stats): idx for idx, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            summaries[futures[future]] = future.result()

    print("  [MR] REDUCE...")
    return reduce_summaries([s for s in summaries if s], stats, strict=strict)


def build_judge_packet(reviews: list[dict], summary: dict) -> str:
    lines = ["## Рекомендации (оцениваем)"]
    for index, action in enumerate(summary.get("action_items", []), 1):
        lines.append(f"  {index}. {action}")

    lines.append("\n## Проблемы пользователей (issues из IE)")
    for review in reviews:
        for issue in review.get("issues", []):
            lines.append(
                f"  - [{review['author']}/{issue['category']}, sev={issue['severity']}] "
                f"«{issue['quote']}»"
            )
    return "\n".join(lines)


def judge_reviews(reviews: list[dict], summary: dict, stats: PipelineStats) -> JudgeReport:
    packet = build_judge_packet(reviews, summary)
    return call_model(
        JudgeReport,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": packet},
        ],
        stats=stats,
    )


def process_one_source(path: Path, stats: PipelineStats) -> dict:
    text = path.read_text(encoding="utf-8")
    return {
        "source": path.stem,
        "aspects": extract_aspects(text, stats),
        "summary": summarize_reviews(text, stats),
    }


def aggregate_multi_doc_aspects(docs: list[dict]) -> pd.DataFrame:
    rows = []
    for doc in docs:
        for review in doc["aspects"]:
            for aspect in review.aspects:
                rows.append(
                    {
                        "source": doc["source"],
                        "author": review.author,
                        "aspect": aspect.aspect,
                        "sentiment": aspect.sentiment,
                        "confidence": aspect.confidence,
                        "quote": aspect.quote,
                    }
                )
    return pd.DataFrame(rows)


def consolidate_multi_doc(
    summaries: list[ReviewsSummary],
    sources: list[str],
    stats: PipelineStats,
) -> MultiDocSummary:
    joined = "\n\n".join(
        f"## {source}\n**Заголовок:** {summary.headline}\n"
        + "\n".join(f"- {item}" for item in summary.key_findings)
        for source, summary in zip(sources, summaries)
    )
    return call_model(
        MultiDocSummary,
        [
            {"role": "system", "content": MULTI_DOC_SYSTEM},
            {"role": "user", "content": joined},
        ],
        stats=stats,
    )


def run_multi_doc(input_dir: Path, out: Path, stats: PipelineStats) -> dict:
    paths = list_multi_doc_sources(input_dir)
    print(f"→ Multi-doc: {len(paths)} источников...")

    with ThreadPoolExecutor(max_workers=4) as pool:
        docs = list(pool.map(lambda path: process_one_source(path, stats), paths))

    df = aggregate_multi_doc_aspects(docs)
    df.to_csv(out / "multi_doc_aspects.csv", index=False, encoding="utf-8")

    pivot = pd.crosstab(df["source"], df["aspect"])
    pivot.to_csv(out / "cross_source_pivot.csv", encoding="utf-8")

    top_topics = df["aspect"].value_counts().head(10)
    top_topics.to_csv(out / "top_topics.csv", header=["count"], encoding="utf-8")

    multi = consolidate_multi_doc(
        [doc["summary"] for doc in docs],
        [doc["source"] for doc in docs],
        stats,
    )
    (out / "multi_doc_summary.json").write_text(
        multi.model_dump_json(indent=2), encoding="utf-8"
    )

    per_source = {
        doc["source"]: {
            "reviews": len(split_by_review(paths[i].read_text(encoding="utf-8"))),
            "headline": doc["summary"].headline,
        }
        for i, doc in enumerate(docs)
    }
    print(f"   сводная таблица: {len(df)} строк, pivot {pivot.shape}")
    return {
        "sources": len(paths),
        "aspect_rows": len(df),
        "top_topic": top_topics.index[0] if len(top_topics) else None,
        "per_source": per_source,
        "multi_doc_headline": multi.overall_headline,
    }


def analyze(input_path: str, out_dir: str = "output") -> PipelineStats:
    stats = PipelineStats()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_ref = Path(input_path)
    input_dir = input_ref if input_ref.is_dir() else input_ref.parent
    corpus = load_corpus(input_ref)
    n_chunks = len(split_by_review(corpus))
    print(f"Загружено: {len(corpus)} символов, {n_chunks} отзывов\n")

    multi_doc_info = run_multi_doc(input_dir, out, stats)
    print()

    print("→ Шаг 1: Information Extraction...")
    reviews = extract_reviews(corpus, stats)
    reviews_data = [item.model_dump(mode="json") for item in reviews]
    (out / "reviews.json").write_text(
        json.dumps(reviews_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"   {len(reviews)} отзывов, {sum(len(r.issues) for r in reviews)} issues")

    print("→ Шаг 2a: аспектный анализ (фиксированный Literal)...")
    aspects = extract_aspects(corpus, stats)
    ghosts = check_quotes_generic(aspects, corpus)
    stats.ghost_quotes = ghosts
    (out / "aspects.json").write_text(
        json.dumps([a.model_dump() for a in aspects], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_heatmap(aspects, out / "heatmap.png")
    print(f"   ghost-цитат (fixed): {len(ghosts)}")

    print("→ Шаг 2b: autodiscovery аспектов...")
    discovered_schema = discover_aspects(corpus, stats)
    (out / "discovered_aspects.json").write_text(
        discovered_schema.model_dump_json(indent=2), encoding="utf-8"
    )
    aspects_discovered = extract_with_discovered(corpus, discovered_schema, stats)
    ghosts_disc = check_quotes_generic(aspects_discovered, corpus)
    stats.ghost_quotes_discovered = ghosts_disc
    (out / "aspects_discovered.json").write_text(
        json.dumps([a.model_dump() for a in aspects_discovered], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison = compare_aspect_sets(aspects, aspects_discovered, discovered_schema)
    (out / "aspect_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"   обнаружено тем: {len(discovered_schema.aspects)}, "
        f"новых вне Literal: {comparison['invented_not_in_literal']}"
    )
    print(f"   ghost-цитат (discovered): {len(ghosts_disc)}")

    print("→ Шаг 3: Map-Reduce...")
    summary = summarize_reviews(corpus, stats)
    (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    print("→ Шаг 4: LLM-as-judge...")
    report = judge_reviews(reviews_data, json.loads(summary.model_dump_json()), stats)

    if report.overall_score < 0.7:
        print(f"   оценка {report.overall_score:.2f} < 0.7 — перезапускаю REDUCE...")
        stats.judge_retries += 1
        summary = summarize_reviews(corpus, stats, strict=True)
        (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        report = judge_reviews(reviews_data, json.loads(summary.model_dump_json()), stats)

    (out / "judge_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    total_aspect_quotes = sum(len(a.aspects) for a in aspects)
    metrics = {
        "input_reviews": n_chunks,
        "extracted_reviews": len(reviews),
        "validation_errors": stats.validation_errors,
        "ghost_quotes_fixed": len(stats.ghost_quotes),
        "ghost_quotes_discovered": len(stats.ghost_quotes_discovered),
        "ghost_quote_share": len(stats.ghost_quotes) / max(1, total_aspect_quotes),
        "overall_score": report.overall_score,
        "elapsed_sec": round(stats.elapsed_sec, 1),
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "estimated_cost_usd": round(stats.estimated_cost_usd, 4),
        "judge_retries": stats.judge_retries,
        "multi_doc": multi_doc_info,
        "aspect_comparison": comparison,
        "judge_verdicts": {
            label: sum(1 for item in report.verdicts if item.support == label)
            for label in ("supported", "weakly_supported", "not_supported")
        },
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ИТОГ ===")
    print(summary.headline)
    print(f"оценка судьи: {report.overall_score:.2f}")
    print(f"multi-doc: {multi_doc_info['multi_doc_headline']}")
    print(f"autodiscovery вне Literal: {comparison['invented_not_in_literal']}")
    print(f"время: {stats.elapsed_sec:.1f} с, ~${stats.estimated_cost_usd:.4f}")
    print(f"артефакты: {out}/")
    return stats


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python pipeline.py <input/|input.txt> [output/]")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")


if __name__ == "__main__":
    main()
