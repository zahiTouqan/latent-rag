"""
evaluate.py - Evaluate the persisted-index RAG baseline on a local QA dataset.

Usage:
    python3 evaluate.py --index_dir artifacts/index --eval_path data/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from metrics import exact_match, recall_at_k, token_f1
from pipeline import (
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TOP_K,
    RAGPipeline,
)

QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
RELEVANT_IDS_FIELD = "relevant_ids"


def _check_bertscore() -> bool:
    try:
        import bert_score  # noqa: F401
        return True
    except ImportError:
        return False


def _read_eval_records(eval_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with eval_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def _normalise_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return [value] if value else []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return []


def load_eval_examples(
    eval_path: Path,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    records = _read_eval_records(eval_path)
    if max_samples is not None:
        records = records[:max_samples]

    examples: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        question_raw = record.get(QUESTION_FIELD)
        question = str(question_raw).strip() if question_raw is not None else ""
        answers = _normalise_list(record.get(ANSWER_FIELD))
        relevant_ids = _normalise_list(record.get(RELEVANT_IDS_FIELD))

        if not question or not answers:
            skipped += 1
            continue

        examples.append(
            {
                "query": question,
                "answers": answers,
                "relevant_ids": relevant_ids,
            }
        )

    if skipped:
        print(f"Warning: skipped {skipped} eval records with missing question or answers.")
    return examples


def aggregate_metrics(
    records: list[dict[str, float]],
    per_query_latency_ms: list[float],
    recall_values: list[float],
    top_k: int,
) -> dict[str, float]:
    if not records:
        raise ValueError("No evaluation records were produced.")

    aggregate = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
    aggregate["latency_p50_ms"] = float(np.percentile(per_query_latency_ms, 50))
    aggregate["latency_p95_ms"] = float(np.percentile(per_query_latency_ms, 95))
    if recall_values:
        aggregate[f"recall@{top_k}"] = float(np.mean(recall_values))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", required=True, help="Local .jsonl or .json QA dataset")
    parser.add_argument(
        "--retrieval_id_field",
        choices=("source_doc_id", "passage_id"),
        default="source_doc_id",
        help="Which retrieved ID type to compare against relevant_ids for recall@k",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--generator_model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--bertscore", action="store_true")
    args = parser.parse_args()

    use_bertscore = args.bertscore
    if use_bertscore and not _check_bertscore():
        print("Warning: bert-score is not installed. Disabling BERTScore. Install with: pip install bert-score")
        use_bertscore = False

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    samples = load_eval_examples(
        eval_path=eval_path,
        max_samples=args.max_samples,
    )
    if not samples:
        raise ValueError(f"No valid evaluation samples were loaded from {eval_path}.")
    print(f"Loaded {len(samples)} samples from {eval_path}")

    pipeline = RAGPipeline(
        index_dir=index_dir,
        seed=args.seed,
        generator_model=args.generator_model,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        f"Loaded index with {pipeline.index_config.passage_count} passages "
        f"from {pipeline.index_config.corpus_path}"
    )

    use_passage_ids = args.retrieval_id_field == "passage_id"

    metric_rows: list[dict[str, float]] = []
    example_rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    references: list[list[str]] = []
    per_query_latency_ms: list[float] = []
    recall_values: list[float] = []

    for start_idx in tqdm(range(0, len(samples), args.eval_batch_size)):
        batch_samples = samples[start_idx : start_idx + args.eval_batch_size]
        queries = [s["query"] for s in batch_samples]
        
        batch_results = pipeline.run_batch(queries)
        
        for sample, result in zip(batch_samples, batch_results):
            per_query_latency_ms.append(result.total_time_s * 1000.0)

            retrieved_ids = result.retrieved_passage_ids if use_passage_ids else list(dict.fromkeys(result.retrieved_source_doc_ids))
            metric_row = {
                "em": exact_match(result.answer, sample["answers"]),
                "f1": token_f1(result.answer, sample["answers"]),
                "retrieval_time_s": result.retrieval_time_s,
                "generation_time_s": result.generation_time_s,
                "total_time_s": result.total_time_s,
                "generated_tokens": result.generated_tokens,
            }

            recall_value = None
            if sample["relevant_ids"]:
                recall_value = recall_at_k(retrieved_ids, sample["relevant_ids"], args.top_k)
                recall_values.append(recall_value)

            example_rows.append(
                {
                    "query": sample["query"],
                    "gold_answers": sample["answers"],
                    "prediction": result.answer,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": sample["relevant_ids"],
                    "recall_at_k": recall_value,
                    **metric_row,
                }
            )
            metric_rows.append(metric_row)
            predictions.append(result.answer)
            references.append(sample["answers"])

    aggregate = aggregate_metrics(metric_rows, per_query_latency_ms, recall_values, args.top_k)
    if use_bertscore:
        from metrics import bertscore

        print("Computing BERTScore...")
        aggregate.update(bertscore(predictions, references))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"results_{eval_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = {
        "config": {
            "eval_path": str(eval_path),
            "retrieval_id_field": args.retrieval_id_field,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "index_dir": str(index_dir),
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "eval_batch_size": args.eval_batch_size,
            "generator_model": args.generator_model,
            "index": asdict(pipeline.index_config),
            "bertscore_enabled": use_bertscore,
        },
        "metrics": aggregate,
        "examples": example_rows,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== Results ===")
    for key, value in sorted(aggregate.items()):
        print(f"  {key:<35s} {value:.4f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
