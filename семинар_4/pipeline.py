"""
RAG по корпусу «Сфера Банк»: ChromaDB + гибрид BM25/dense (RRF) + структурированный ответ.

Сравнение чанкинга (ДЗ семинар 4):
  fixed     — text[i:i+2000], без перекрытия
  recursive — RecursiveCharacterTextSplitter(400, overlap=80)

Команды:
    python pipeline.py ingest --chunking fixed
    python pipeline.py ingest --chunking recursive
    python pipeline.py ask "Кто жаловался на push-уведомления?"
    python eval.py --compare
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_client import get_model, make_client
from rank_bm25 import BM25Okapi
from schema import RAGAnswer

client = make_client()
MODEL = get_model()

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
BM25_CACHE = Path(__file__).parent / "bm25_cache.json"
CHUNKING_FILE = Path(__file__).parent / "output" / "chunking_strategy.txt"

EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)

RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "],
)

_chroma = None
_collection = None
_active_chunking = "recursive"


def get_chroma():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma


def get_collection(chunking: str | None = None):
    """Отдельная коллекция на стратегию чанкинга — можно сравнивать без перезаписи."""
    global _collection, _active_chunking
    strategy = chunking or _read_active_chunking()
    if _collection is None or strategy != _active_chunking:
        _active_chunking = strategy
        name = f"sfera_bank_{strategy}"
        _collection = get_chroma().get_or_create_collection(
            name=name,
            embedding_function=EMBED_FN,
            metadata={"hnsw:space": "cosine", "chunking": strategy},
        )
    return _collection


# eval.py импортирует collection — проксируем на активную стратегию
class _CollectionProxy:
    def __getattr__(self, name):
        return getattr(get_collection(), name)


collection = _CollectionProxy()


def _read_active_chunking() -> str:
    if CHUNKING_FILE.exists():
        return CHUNKING_FILE.read_text(encoding="utf-8").strip()
    return os.environ.get("CHUNKING_STRATEGY", "recursive")


def _save_active_chunking(strategy: str) -> None:
    CHUNKING_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKING_FILE.write_text(strategy, encoding="utf-8")


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def chunk_text_fixed(text: str, chunk_size: int = 2000) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size) if text[i : i + chunk_size].strip()]


def chunk_text_recursive(text: str) -> list[str]:
    return [c.strip() for c in RECURSIVE_SPLITTER.split_text(text) if c.strip()]


def chunk_document(text: str, strategy: str) -> list[str]:
    if strategy == "fixed":
        return chunk_text_fixed(text)
    if strategy == "recursive":
        return chunk_text_recursive(text)
    raise ValueError(f"Неизвестная стратегия: {strategy}")


def ingest(chunking: str = "recursive") -> dict:
    global _collection
    _collection = None
    _save_active_chunking(chunking)
    col = get_collection(chunking)

    existing = col.get()
    if existing["ids"]:
        col.delete(ids=existing["ids"])

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []
    per_file: dict[str, int] = {}

    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_document(text, chunking)
        per_file[f.stem] = len(chunks)

        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"source": f.stem, "chunk_id": i, "chunking": chunking})

        print(f"  {f.stem}: {len(chunks)} чанков")

    col.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)

    cache_path = BM25_CACHE.with_name(f"bm25_cache_{chunking}.json")
    cache_path.write_text(
        json.dumps({"ids": all_ids, "tokens": [tokenize_ru(c) for c in all_chunks], "texts": all_chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    BM25_CACHE.write_text(cache_path.read_text(encoding="utf-8"), encoding="utf-8")

    stats = {
        "chunking": chunking,
        "files": len(per_file),
        "chunks": len(all_chunks),
        "avg_chunks_per_file": round(len(all_chunks) / max(1, len(per_file)), 1),
        "per_file": per_file,
    }
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    (out / f"ingest_{chunking}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nИндекс [{chunking}]: {stats['chunks']} чанков из {stats['files']} файлов")
    print(f"BM25 кэш: {cache_path.name}")
    return stats


def _load_bm25(chunking: str | None = None):
    strategy = chunking or _read_active_chunking()
    cache_path = BM25_CACHE.with_name(f"bm25_cache_{strategy}.json")
    if not cache_path.exists():
        cache_path = BM25_CACHE
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return BM25Okapi(data["tokens"]), data["ids"], data["texts"]


def hybrid_retrieve(query: str, k: int = 5, top: int = 15, c: int = 60, chunking: str | None = None) -> dict:
    col = get_collection(chunking)
    dense = col.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    bm25, bm25_ids, bm25_texts = _load_bm25(chunking)
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)
    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    rrf: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank + 1)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return {"ids": [top_ids], "documents": [[text_by_id[i] for i in top_ids]]}


def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос продакта по внутреннему архиву «Сфера Банк». "
        "Опирайся ТОЛЬКО на контекст ниже. Если в контексте нет ответа — скажи прямо.\n\n"
        "Правила:\n"
        "1. Только факты из контекста, без общих знаний.\n"
        "2. В quotes — 1-5 точных коротких цитат (не пересказ).\n"
        "3. В sources — id блоков (формат: 'interview_ivan__2').\n"
        "4. confidence: 0.9+ если прямой ответ в контексте, 0.5-0.8 если собран из кусков, "
        "< 0.5 если контекст не отвечает.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


def ask(query: str, chunking: str | None = None):
    strategy = chunking or _read_active_chunking()
    _save_active_chunking(strategy)
    global _collection
    _collection = None

    print(f"Стратегия чанкинга: {strategy}")
    print("Поиск по базе...", flush=True)
    t0 = time.time()
    hits = hybrid_retrieve(query, k=5, chunking=strategy)
    found = hits["ids"][0]
    print(f"   {len(found)} чанков за {time.time() - t0:.1f}с: {', '.join(found)}", flush=True)

    print("Генерация ответа...", flush=True)
    t1 = time.time()
    prompt = build_prompt(query, hits)
    resp: RAGAnswer = client.chat.completions.create(
        model=MODEL,
        response_model=RAGAnswer,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_retries=3,
    )
    print(f"   ответ за {time.time() - t1:.1f}с", flush=True)

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    print(resp.model_dump_json(indent=2, ensure_ascii=False))
    print("\n--- источники ---")
    for i in found:
        print(f"  {i}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--chunking", choices=["fixed", "recursive"], default="recursive")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--chunking", choices=["fixed", "recursive"], default=None)

    args = parser.parse_args() if len(sys.argv) > 1 else None

    if args is None or args.cmd is None:
        print("Использование: python pipeline.py {ingest|ask} ...")
        sys.exit(1)

    if args.cmd == "ingest":
        print(f"Загружаю эмбеддер и индексирую [{args.chunking}]...", flush=True)
        ingest(args.chunking)
    elif args.cmd == "ask":
        ask(args.question, args.chunking)


if __name__ == "__main__":
    main()
