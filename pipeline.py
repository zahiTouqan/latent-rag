"""
pipeline.py - Simple persisted-index RAG baseline.

Offline:
    passage JSONL -> embeddings -> FAISS index + metadata

Online:
    query -> dense retrieval -> prompt -> generation
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_TOP_K = 5
DEFAULT_MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

PROMPT_PREFIX = """Answer the following question using only the provided context.
If the answer is not supported by the context, say you do not know.

Context:
"""
PROMPT_SUFFIX_TEMPLATE = """

Question: {question}

Answer:"""


@dataclass
class Passage:
    passage_id: str
    source_doc_id: str
    text: str


@dataclass
class Result:
    query: str
    answer: str
    retrieved_passage_ids: list[str]
    retrieved_source_doc_ids: list[str]
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float
    generated_tokens: int


@dataclass
class IndexConfig:
    embedding_model: str
    corpus_path: str
    passage_count: int
    max_docs: int | None = None
    doc_id_provided_count: int | None = None
    doc_id_missing_count: int | None = None


def load_index_config(index_dir: str | Path) -> IndexConfig:
    config_path = Path(index_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Index config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return IndexConfig(**json.load(handle))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generator_dtype() -> torch.dtype:
    if DEVICE == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _load_to_device(model: torch.nn.Module, label: str) -> torch.nn.Module:
    """Move *model* to DEVICE, falling back to CPU on OOM."""
    if DEVICE != "cuda":
        return model.to("cpu")
    try:
        return model.to(DEVICE)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"Warning: GPU OOM when loading {label}. Falling back to CPU.")
        return model.to("cpu")


class Retriever:
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        print(f"Loading embedding model: {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        dtype = _generator_dtype() if DEVICE == "cuda" else torch.float32
        self.model = _load_to_device(
            AutoModel.from_pretrained(embedding_model, torch_dtype=dtype),
            "embedding model",
        ).eval()
        self.index: faiss.IndexFlatIP | None = None
        self.passages: list[Passage] = []

    @torch.no_grad()
    def _embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            texts = ["Represent this sentence for searching relevant passages: " + text for text in texts]
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1).cpu().float().numpy()

    def build_index(self, passages: list[Passage], batch_size: int = 256) -> None:
        if not passages:
            raise ValueError("Cannot build an index with zero passages.")

        self.passages = passages
        n_batches = (len(passages) + batch_size - 1) // batch_size
        print(f"Embedding {len(passages)} passages ({n_batches} batches)...")
        all_embeddings = []
        for i, start in enumerate(range(0, len(passages), batch_size)):
            batch = passages[start : start + batch_size]
            all_embeddings.append(self._embed([passage.text for passage in batch]))
            if (i + 1) % 25 == 0 or i + 1 == n_batches:
                print(f"  batch {i + 1}/{n_batches}")
        embeddings = np.concatenate(all_embeddings, axis=0)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def save(
        self,
        index_dir: str | Path,
        corpus_path: str,
        max_docs: int | None = None,
        doc_id_provided_count: int | None = None,
        doc_id_missing_count: int | None = None,
    ) -> None:
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for passage in self.passages:
                handle.write(json.dumps(asdict(passage), ensure_ascii=True) + "\n")

        config = IndexConfig(
            embedding_model=self.embedding_model,
            corpus_path=str(corpus_path),
            passage_count=len(self.passages),
            max_docs=max_docs,
            doc_id_provided_count=doc_id_provided_count,
            doc_id_missing_count=doc_id_missing_count,
        )
        with (index_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)

    def load(self, index_dir: str | Path) -> IndexConfig:
        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "metadata.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.passages = []
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.passages.append(Passage(**record))

        config = load_index_config(index_dir)

        if self.index.ntotal != len(self.passages):
            raise ValueError(
                "Index and metadata size mismatch: "
                f"index has {self.index.ntotal} rows but metadata has {len(self.passages)} passages."
            )
        return config

    def retrieve(self, query: str, top_k: int) -> tuple[list[Passage], float]:
        if self.index is None:
            raise ValueError("Retriever index is not loaded.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        t0 = time.perf_counter()
        query_embedding = self._embed([query], is_query=True)
        _, indices = self.index.search(query_embedding, min(top_k, len(self.passages)))
        elapsed = time.perf_counter() - t0

        passages = [self.passages[idx] for idx in indices[0] if idx != -1]
        return passages, elapsed


class Generator:
    def __init__(
        self,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.generator_model = generator_model
        self.max_new_tokens = max_new_tokens

        print(f"Loading generator model: {generator_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = _load_to_device(
            AutoModelForCausalLM.from_pretrained(
                generator_model,
                torch_dtype=_generator_dtype(),
            ),
            "generator model",
        ).eval()

    def _render_prompt(self, query: str, passages: list[Passage]) -> str:
        context_blocks = [f"[{idx}] {passage.text.strip()}" for idx, passage in enumerate(passages, start=1)]
        context = "\n\n".join(context_blocks)
        return f"{PROMPT_PREFIX}{context}{PROMPT_SUFFIX_TEMPLATE.format(question=query)}"

    @torch.no_grad()
    def generate(self, query: str, passages: list[Passage]) -> tuple[str, dict]:
        prompt = self._render_prompt(query, passages)
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
        prompt_tokens = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generation_time = time.perf_counter() - t0

        new_tokens = output_ids[0, prompt_tokens:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        stats = {
            "generated_tokens": int(new_tokens.shape[-1]),
            "generation_time_s": generation_time,
        }
        return answer, stats


class RAGPipeline:
    def __init__(
        self,
        index_dir: str | Path,
        seed: int = SEED,
        embedding_model: str | None = None,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
        top_k: int = DEFAULT_TOP_K,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        set_seed(seed)
        self.top_k = top_k
        self.index_config = load_index_config(index_dir)
        actual_embedding_model = embedding_model or self.index_config.embedding_model
        if actual_embedding_model != self.index_config.embedding_model:
            raise ValueError(
                "Requested embedding model does not match the loaded index. "
                f"Requested {actual_embedding_model}, index was built with {self.index_config.embedding_model}."
            )
        self.retriever = Retriever(actual_embedding_model)
        self.retriever.load(index_dir)
        self.generator = Generator(
            generator_model=generator_model,
            max_new_tokens=max_new_tokens,
        )

    def run(self, query: str) -> Result:
        t0 = time.perf_counter()
        retrieved_passages, retrieval_time = self.retriever.retrieve(query, self.top_k)
        answer, stats = self.generator.generate(query, retrieved_passages)

        return Result(
            query=query,
            answer=answer,
            retrieved_passage_ids=[p.passage_id for p in retrieved_passages],
            retrieved_source_doc_ids=[p.source_doc_id for p in retrieved_passages],
            retrieval_time_s=retrieval_time,
            generation_time_s=stats["generation_time_s"],
            total_time_s=time.perf_counter() - t0,
            generated_tokens=stats["generated_tokens"],
        )
