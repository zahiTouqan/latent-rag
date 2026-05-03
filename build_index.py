"""
build_index.py - Build and persist a FAISS passage index.

Usage:
    python3 build_index.py --corpus_path data/passages.jsonl --index_dir artifacts/index --retriever_type bge
    python3 build_index.py --corpus_path data/passages.jsonl --index_dir artifacts/index_latent --retriever_type latent
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pipeline import (
    DEFAULT_BGE_MODEL,
    DEFAULT_LATENT_MODEL,
    BGERTRetriever,
    LatentRetriever,
    Passage,
)


@dataclass
class LoadStats:
    skipped: int = 0
    doc_id_provided_count: int = 0
    doc_id_missing_count: int = 0


def load_passages(corpus_path: Path, max_docs: int | None) -> tuple[list[Passage], LoadStats]:
    passages: list[Passage] = []
    seen_passage_ids: set[str] = set()
    stats = LoadStats()

    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_docs is not None and len(passages) >= max_docs:
                break

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            passage_id = str(record.get("id", "")).strip()
            text = str(record.get("text", "")).strip()
            raw_doc_id = record.get("doc_id")
            source_doc_id = str(raw_doc_id).strip() if raw_doc_id is not None else ""

            if not passage_id or not text:
                stats.skipped += 1
                continue
            if passage_id in seen_passage_ids:
                raise ValueError(
                    f"Duplicate passage id '{passage_id}' at line {line_number}. "
                    "Passage ids must be unique. Use doc_id for article provenance and id for passage ids."
                )

            if source_doc_id:
                stats.doc_id_provided_count += 1
            else:
                stats.doc_id_missing_count += 1
                source_doc_id = passage_id

            seen_passage_ids.add(passage_id)
            passages.append(Passage(passage_id=passage_id, source_doc_id=source_doc_id, text=text))

    if stats.skipped:
        print(f"Warning: skipped {stats.skipped} passages with missing 'id' or 'text' fields.")

    return passages, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True, help="Passage JSONL with fields id, text, and optional doc_id")
    parser.add_argument("--index_dir", default="artifacts/index")
    parser.add_argument(
        "--retriever_type",
        choices=("bge", "latent"),
        default="bge",
        help="Retriever type to build index for",
    )
    parser.add_argument("--embedding_model", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    args = parser.parse_args()

    if args.embedding_model is None:
        args.embedding_model = DEFAULT_BGE_MODEL if args.retriever_type == "bge" else DEFAULT_LATENT_MODEL
    if args.batch_size is None:
        args.batch_size = 256 if args.retriever_type == "bge" else 64

    corpus_path = Path(args.corpus_path)
    if corpus_path.suffix.lower() != ".jsonl":
        raise ValueError("Only .jsonl corpora are supported.")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    passages, stats = load_passages(corpus_path, args.max_docs)
    if not passages:
        raise ValueError("No valid passages were loaded from the corpus.")

    print(f"Loaded {len(passages)} passages from {corpus_path}")
    if stats.doc_id_missing_count:
        print(
            f"Warning: {stats.doc_id_missing_count} passages are missing doc_id. "
            "Document-level recall may be inaccurate unless the corpus provides explicit doc_id values."
        )

    if args.retriever_type == "bge":
        retriever = BGERTRetriever(embedding_model=args.embedding_model)
    else:
        retriever = LatentRetriever(embedding_model=args.embedding_model)

    retriever.build_index(passages, batch_size=args.batch_size)
    retriever.save(
        index_dir=args.index_dir,
        corpus_path=str(corpus_path),
        max_docs=args.max_docs,
        doc_id_provided_count=stats.doc_id_provided_count,
        doc_id_missing_count=stats.doc_id_missing_count,
    )
    print(f"Saved {args.retriever_type} index to {args.index_dir}")


if __name__ == "__main__":
    main()
