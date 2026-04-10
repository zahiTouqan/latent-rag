# latent-rag

Minimal traditional RAG baseline for local corpora and local QA datasets.

## What this repo does

- Loads a passage corpus from JSONL
- Builds a dense FAISS index once and saves it to disk
- Retrieves top-k passages with BGE embeddings
- Generates answers with a causal language model
- Evaluates on a local JSONL QA file
- Reports answer quality, optional retrieval recall, and per-query latency
- Saves both aggregate metrics and per-example outputs

Core files:

- `build_index.py`: load a passage corpus, embed it, and write a persisted FAISS index
- `pipeline.py`: retrieval, prompt construction, generation, and runtime pipeline
- `evaluate.py`: local dataset loading, evaluation loop, and result saving
- `metrics.py`: retrieval and generation metrics helpers

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional:

```bash
pip install bert-score
```

## Corpus format

JSONL only: one passage per line.

Required fields:

- `id`: unique passage ID
- `text`: passage text

Optional field:

- `doc_id`: source document ID if you want document-level recall

Example passage corpus:

```json
{"id": "doc-12:0", "doc_id": "doc-12", "text": "First passage from document 12 ..."}
{"id": "doc-12:1", "doc_id": "doc-12", "text": "Second passage from document 12 ..."}
{"id": "doc-13:0", "doc_id": "doc-13", "text": "Passage from document 13 ..."}
```

Notes:

- Passage `id` values must be unique.
- If `doc_id` is omitted, the index falls back to `id`.
- Use `doc_id` if your eval file provides document-level `relevant_ids`.

## Eval dataset format

JSONL only: one record per line.

Required fields:

- `question`: the question text
- `answer`: a string or list of acceptable answer strings

Optional field:

- `relevant_ids`: IDs for recall@k

Example eval JSONL:

```json
{"question": "What is the capital of France?", "answer": "Paris", "relevant_ids": ["doc-42"]}
{"question": "Who wrote Hamlet?", "answer": ["William Shakespeare", "Shakespeare"], "relevant_ids": ["doc-9"]}
```

If `relevant_ids` are document IDs, use `--retrieval_id_field source_doc_id`.
If they are passage IDs, use `--retrieval_id_field passage_id`.

## Build the index

```bash
python3 build_index.py \
  --corpus_path /path/to/passages.jsonl \
  --index_dir artifacts/index
```

Useful options:

- `--max_docs 100000`: stop reading after N passages
- `--embedding_model BAAI/bge-base-en-v1.5`
- `--batch_size 256`

The index directory will contain:

- `index.faiss`
- `metadata.jsonl`
- `config.json`

## Evaluate

Smoke test:

```bash
python3 evaluate.py \
  --eval_path /path/to/eval.jsonl \
  --max_samples 20 \
  --index_dir artifacts/index
```

Document-level recall:

```bash
python3 evaluate.py \
  --eval_path /path/to/eval.jsonl \
  --index_dir artifacts/index \
  --retrieval_id_field source_doc_id
```

Passage-level recall:

```bash
python3 evaluate.py \
  --eval_path /path/to/eval.jsonl \
  --index_dir artifacts/index \
  --retrieval_id_field passage_id
```

Useful options:

- `--top_k 5`
- `--max_new_tokens 128`
- `--generator_model Qwen/Qwen3.5-0.8B`
- `--bertscore`

## Metrics reported

- End-to-end quality: `em`, `f1`, optional `bertscore_*`
- Retrieval: `recall@k` when the eval file includes `relevant_ids`
- Efficiency: mean latency fields, `latency_p50_ms`, `latency_p95_ms`

## Results

Results are written to `results/results_<eval_file_stem>_<timestamp>.json`.

Each result file contains:

- `config`: run settings and index metadata
- `metrics`: aggregated metrics
- `examples`: per-query predictions, retrieved IDs, and timings

## Colab

Use [baseline_nq_colab.ipynb](/home/zahi/projects/latent-rag/baseline_nq_colab.ipynb) as a thin generic runner notebook.

The notebook streams a configurable subset of `facebook/kilt_wikipedia`, loads Google Natural Questions from `google-research-datasets/natural_questions`, converts it into the local eval format, and evaluates against document-level relevance labels without requiring a full Wikipedia download in Colab.
