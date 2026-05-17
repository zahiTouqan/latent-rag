"""
evaluate.py - Evaluate RAG pipeline on a local QA dataset.

Usage:
    python3 evaluate.py --index_dir artifacts/index_bge --eval_path data/eval.jsonl --mode bge+text
    python3 evaluate.py --index_dir artifacts/index_bge --eval_path data/eval.jsonl --mode bge+latent
    python3 evaluate.py --index_dir artifacts/index_bge --eval_path data/eval.jsonl --mode bge+t5text
    python3 evaluate.py --index_dir artifacts/index_latent --eval_path data/eval.jsonl --mode t5+text
    python3 evaluate.py --index_dir artifacts/index_latent --eval_path data/eval.jsonl --mode t5+latent
    python3 evaluate.py --index_dir artifacts/index_latent --eval_path data/eval.jsonl --mode t5+t5text
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from tqdm import tqdm

from metrics import exact_match, normalise_answer, recall_at_k, token_f1
from pipeline import (
    DEFAULT_TEXT_GENERATOR,
    DEFAULT_LATENT_GENERATOR,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TOP_K,
    BGERTRetriever,
    LatentGenerator,
    LatentRetriever,
    RAGPipeline,
    Seq2SeqTextGenerator,
    TextGenerator,
    load_index_config,
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


def answer_present_in_context(contexts: list[str], answers: list[str]) -> float:
    normalised_context = normalise_answer("\n".join(contexts))
    for answer in answers:
        answer_text = normalise_answer(answer)
        if answer_text and answer_text in normalised_context:
            return 1.0
    return 0.0


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


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
        relevant_ids = list(dict.fromkeys(_normalise_list(record.get(RELEVANT_IDS_FIELD))))

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


def build_pipeline(
    mode: str,
    index_dir: str | Path,
    generator_model: str | None,
    top_k: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[RAGPipeline, IndexConfig]:
    index_config = load_index_config(index_dir)

    if mode.startswith("bge"):
        retriever = BGERTRetriever(embedding_model=index_config.embedding_model)
        retriever.load(index_dir)
    elif mode.startswith("t5"):
        retriever = LatentRetriever(embedding_model=index_config.embedding_model)
        retriever.load(index_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode.endswith("+t5text"):
        gen_model = generator_model or DEFAULT_LATENT_GENERATOR
        generator = Seq2SeqTextGenerator(generator_model=gen_model, max_new_tokens=max_new_tokens)
    elif mode.endswith("+text"):
        gen_model = generator_model or DEFAULT_TEXT_GENERATOR
        generator = TextGenerator(generator_model=gen_model, max_new_tokens=max_new_tokens)
    elif mode.endswith("+latent"):
        gen_model = generator_model or (index_config.embedding_model if mode == "t5+latent" else DEFAULT_LATENT_GENERATOR)
        if mode == "t5+latent" and gen_model != index_config.embedding_model:
            raise ValueError(
                "t5+latent generation requires the generator model to match the latent index embedding model. "
                f"Index was built with {index_config.embedding_model}, but generator_model is {gen_model}."
            )
        generator = LatentGenerator(generator_model=gen_model, max_new_tokens=max_new_tokens)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pipeline = RAGPipeline(retriever=retriever, generator=generator, top_k=top_k, seed=seed)
    return pipeline, index_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", required=True, help="Local .jsonl QA dataset")
    parser.add_argument(
        "--mode",
        choices=("bge+text", "bge+latent", "bge+t5text", "t5+text", "t5+latent", "t5+t5text"),
        default="bge+text",
        help="Pipeline mode: retriever+generator combination",
    )
    parser.add_argument(
        "--retrieval_id_field",
        choices=("source_doc_id", "passage_id"),
        default="source_doc_id",
        help="Which retrieved ID type to compare against relevant_ids for recall@k",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--generator_model", default=None)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--warmup", action="store_true", help="Run one untimed warm-up query before evaluation")
    parser.add_argument("--quality_gate", action="store_true", help="Fail fast on invalid generations during smoke tests")
    parser.add_argument("--max_visible_special_tokens", type=int, default=0)
    parser.add_argument("--max_repeated_3gram", type=int, default=3)
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

    pipeline, index_config = build_pipeline(
        mode=args.mode,
        index_dir=index_dir,
        generator_model=args.generator_model,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(
        f"Pipeline mode: {args.mode} | "
        f"Index: {index_config.passage_count} passages from {index_config.corpus_path}"
    )
    if args.warmup:
        print("Running untimed warm-up query...")
        pipeline.run(samples[0]["query"])

    use_passage_ids = args.retrieval_id_field == "passage_id"

    metric_rows: list[dict[str, float]] = []
    example_rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    references: list[list[str]] = []
    per_query_latency_ms: list[float] = []
    recall_values: list[float] = []
    answer_support_values: list[float] = []

    for sample in tqdm(samples):
        result = pipeline.run(sample["query"])
        per_query_latency_ms.append(result.total_time_s * 1000.0)

        retrieved_ids = (
            result.retrieved_passage_ids
            if use_passage_ids
            else list(dict.fromkeys(result.retrieved_source_doc_ids))
        )
        answer_support_value = answer_present_in_context(result.retrieved_texts, sample["answers"])
        metric_row = {
            "em": exact_match(result.answer, sample["answers"]),
            "f1": token_f1(result.answer, sample["answers"]),
            "retrieval_time_s": result.retrieval_time_s,
            "generation_time_s": result.generation_time_s,
            "total_time_s": result.total_time_s,
            "generated_tokens": result.generated_tokens,
            "ended_with_eos": float(bool(result.generation_diagnostics.get("ended_with_eos", False))),
            "visible_special_tokens": float(result.generation_diagnostics.get("visible_special_tokens", 0)),
            "max_repeated_3gram": float(result.generation_diagnostics.get("max_repeated_3gram", 0)),
        }
        answer_support_values.append(answer_support_value)

        recall_value = None
        if sample["relevant_ids"]:
            recall_value = recall_at_k(retrieved_ids, sample["relevant_ids"], args.top_k)
            recall_values.append(recall_value)

        if args.quality_gate:
            visible_special_tokens = int(result.generation_diagnostics.get("visible_special_tokens", 0))
            max_repeated_3gram = int(result.generation_diagnostics.get("max_repeated_3gram", 0))
            ended_with_eos = bool(result.generation_diagnostics.get("ended_with_eos", False))
            if visible_special_tokens > args.max_visible_special_tokens:
                raise RuntimeError(
                    f"Quality gate failed for query '{sample['query']}': "
                    f"visible_special_tokens={visible_special_tokens}"
                )
            if max_repeated_3gram > args.max_repeated_3gram:
                raise RuntimeError(
                    f"Quality gate failed for query '{sample['query']}': "
                    f"max_repeated_3gram={max_repeated_3gram}"
                )
            if result.generated_tokens >= args.max_new_tokens and not ended_with_eos:
                raise RuntimeError(
                    f"Quality gate failed for query '{sample['query']}': "
                    "generation reached max_new_tokens without EOS"
                )

        example_rows.append(
            {
                "query": sample["query"],
                "gold_answers": sample["answers"],
                "prediction": result.answer,
                "retrieved_ids": retrieved_ids,
                "retrieved_passages": [
                    {
                        "passage_id": passage_id,
                        "source_doc_id": source_doc_id,
                        "score": score,
                        "text": text[:1000],
                    }
                    for passage_id, source_doc_id, score, text in zip(
                        result.retrieved_passage_ids,
                        result.retrieved_source_doc_ids,
                        result.retrieval_scores,
                        result.retrieved_texts,
                    )
                ],
                "relevant_ids": sample["relevant_ids"],
                "recall_at_k": recall_value,
                f"answer_support@{args.top_k}": answer_support_value,
                "generation_diagnostics": result.generation_diagnostics,
                **metric_row,
            }
        )
        metric_rows.append(metric_row)
        predictions.append(result.answer)
        references.append(sample["answers"])

    aggregate = aggregate_metrics(metric_rows, per_query_latency_ms, recall_values, args.top_k)
    aggregate[f"answer_support@{args.top_k}"] = float(np.mean(answer_support_values))
    if use_bertscore:
        from metrics import bertscore

        print("Computing BERTScore...")
        aggregate.update(bertscore(predictions, references))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"results_{eval_path.stem}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    resolved_generator_model = getattr(pipeline.generator, "generator_model", args.generator_model)
    resolved_retriever_model = getattr(pipeline.retriever, "embedding_model", None)
    payload = {
        "config": {
            "eval_path": str(eval_path),
            "mode": args.mode,
            "retrieval_id_field": args.retrieval_id_field,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "index_dir": str(index_dir),
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "generator_model": resolved_generator_model,
            "retriever_model": resolved_retriever_model,
            "generation_kwargs": example_rows[0]["generation_diagnostics"].get("generation_kwargs", {}) if example_rows else {},
            "warmup_enabled": args.warmup,
            "quality_gate_enabled": args.quality_gate,
            "index": asdict(index_config),
            "bertscore_enabled": use_bertscore,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": getattr(next(pipeline.generator.model.parameters()), "device", None).type,
                "dtype": str(next(pipeline.generator.model.parameters()).dtype),
                "git_revision": git_revision(),
            },
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
