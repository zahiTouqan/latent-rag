"""
pipeline.py - Flexible RAG pipeline supporting traditional and latent-space modes.

Retrievers:
    BGERTRetriever   - BGE dense retrieval (AutoModel + [CLS] pooling)
    LatentRetriever  - T5 encoder latent retrieval (FAISS + safetensors)

Generators:
    TextGenerator    - causal LM text generation from prompted passages
    LatentGenerator  - encoder-decoder generation from latent representations

Pipeline:
    RAGPipeline      - generic orchestrator accepting any retriever + generator
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
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

DEFAULT_BGE_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_LATENT_MODEL = "google/t5gemma-2-270m-270m"
DEFAULT_TEXT_GENERATOR = "Qwen/Qwen3.5-0.8B"
DEFAULT_LATENT_GENERATOR = "google/t5gemma-2-270m-270m"
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
    retriever_type: str
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
    if DEVICE != "cuda":
        return model.to("cpu")
    try:
        return model.to(DEVICE)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"Warning: GPU OOM when loading {label}. Falling back to CPU.")
        return model.to("cpu")


# ---------------------------------------------------------------------------
# BGERTRetriever  (traditional dense retrieval with BGE/BERT embeddings)
# ---------------------------------------------------------------------------

class BGERTRetriever:
    def __init__(self, embedding_model: str = DEFAULT_BGE_MODEL):
        self.embedding_model = embedding_model
        print(f"Loading BGE embedding model: {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        dtype = _generator_dtype() if DEVICE == "cuda" else torch.float32
        self.model = _load_to_device(
            AutoModel.from_pretrained(embedding_model, torch_dtype=dtype),
            "BGE embedding model",
        ).eval()
        self.index: faiss.IndexFlatIP | None = None
        self.passages: list[Passage] = []

    @torch.no_grad()
    def _embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
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
            all_embeddings.append(self._embed([p.text for p in batch]))
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
            retriever_type="bge",
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
                f"Index and metadata size mismatch: index has {self.index.ntotal} rows "
                f"but metadata has {len(self.passages)} passages."
            )
        return config

    def retrieve(self, query: str, top_k: int) -> tuple[list[Passage], None, float]:
        if self.index is None:
            raise ValueError("Retriever index is not loaded.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        t0 = time.perf_counter()
        query_embedding = self._embed([query], is_query=True)
        _, indices = self.index.search(query_embedding, min(top_k, len(self.passages)))
        elapsed = time.perf_counter() - t0
        passages = [self.passages[idx] for idx in indices[0] if idx != -1]
        return passages, None, elapsed


# ---------------------------------------------------------------------------
# LatentRetriever  (T5 encoder latent retrieval with safetensors storage)
# ---------------------------------------------------------------------------

class LatentRetriever:
    def __init__(self, embedding_model: str = DEFAULT_LATENT_MODEL):
        self.embedding_model = embedding_model
        print(f"Loading latent embedding model (encoder only): {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        dtype = _generator_dtype() if DEVICE == "cuda" else torch.float32
        full_model = AutoModelForSeq2SeqLM.from_pretrained(embedding_model, torch_dtype=dtype)
        self.model = _load_to_device(full_model.get_encoder(), "latent encoder").eval()
        self.index: faiss.IndexFlatIP | None = None
        self.passages: list[Passage] = []
        self.all_latents: dict[str, torch.Tensor] = {}
        self.latent_file: str | None = None

    @torch.no_grad()
    def _embed(self, texts: list[str]) -> tuple[np.ndarray, list[torch.Tensor]]:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**encoded)
        latents = outputs.last_hidden_state.cpu()
        attention_mask = encoded["attention_mask"].cpu()
        faiss_vecs = []
        saved_latents = []
        for i in range(len(texts)):
            seq_len = attention_mask[i].sum().item()
            true_latents = latents[i, :seq_len, :]
            saved_latents.append(true_latents)
            mean_pooled = true_latents.mean(dim=0).unsqueeze(0)
            faiss_vec = F.normalize(mean_pooled, p=2, dim=-1).float().numpy()
            faiss_vecs.append(faiss_vec)
        return np.concatenate(faiss_vecs, axis=0), saved_latents

    def build_index(self, passages: list[Passage], batch_size: int = 64) -> None:
        if not passages:
            raise ValueError("Cannot build an index with zero passages.")
        self.passages = passages
        n_batches = (len(passages) + batch_size - 1) // batch_size
        print(f"Embedding {len(passages)} passages ({n_batches} batches)...")
        all_embeddings = []
        self.all_latents = {}
        for i, start in enumerate(range(0, len(passages), batch_size)):
            batch = passages[start : start + batch_size]
            faiss_batch, latents_batch = self._embed([p.text for p in batch])
            all_embeddings.append(faiss_batch)
            for p, lat in zip(batch, latents_batch):
                self.all_latents[p.passage_id] = lat
            if (i + 1) % 10 == 0 or i + 1 == n_batches:
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
        print("Saving latent sequences to safetensors...")
        save_file(self.all_latents, str(index_dir / "latents.safetensors"))
        config = IndexConfig(
            retriever_type="latent",
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
        self.latent_file = str(index_dir / "latents.safetensors")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not Path(self.latent_file).exists():
            raise FileNotFoundError(f"Safetensors latents not found: {self.latent_file}")
        self.index = faiss.read_index(str(index_path))
        self.passages = []
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.passages.append(Passage(**record))
        config = load_index_config(index_dir)
        if self.index.ntotal != len(self.passages):
            raise ValueError(
                f"Index and metadata size mismatch: index has {self.index.ntotal} rows "
                f"but metadata has {len(self.passages)} passages."
            )
        return config

    def retrieve(self, query: str, top_k: int) -> tuple[list[Passage], list[torch.Tensor], float]:
        if self.index is None:
            raise ValueError("Retriever index is not loaded.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        t0 = time.perf_counter()
        query_faiss, _ = self._embed([query])
        _, indices = self.index.search(query_faiss, min(top_k, len(self.passages)))
        passages = [self.passages[idx] for idx in indices[0] if idx != -1]
        device = next(self.model.parameters()).device
        latents_list = []
        with safe_open(self.latent_file, framework="pt", device="cpu") as f:
            for p in passages:
                lat = f.get_tensor(p.passage_id).to(device)
                latents_list.append(lat)
        elapsed = time.perf_counter() - t0
        return passages, latents_list, elapsed


# ---------------------------------------------------------------------------
# TextGenerator  (causal LM, text prompt from retrieved passages)
# ---------------------------------------------------------------------------

class TextGenerator:
    def __init__(
        self,
        generator_model: str = DEFAULT_TEXT_GENERATOR,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.generator_model = generator_model
        self.max_new_tokens = max_new_tokens
        print(f"Loading text generator model: {generator_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = _load_to_device(
            AutoModelForCausalLM.from_pretrained(
                generator_model,
                torch_dtype=_generator_dtype(),
            ),
            "text generator",
        ).eval()

    def _render_prompt(self, query: str, passages: list[Passage]) -> str:
        context_blocks = [f"[{idx}] {p.text.strip()}" for idx, p in enumerate(passages, start=1)]
        context = "\n\n".join(context_blocks)
        user_content = (
            f"Answer the following question using only the provided context.\n"
            f"If the answer is not supported by the context, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": user_content}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{PROMPT_PREFIX}{context}{PROMPT_SUFFIX_TEMPLATE.format(question=query)}"

    @torch.no_grad()
    def generate(
        self,
        query: str,
        passages: list[Passage],
        passage_latents: list[torch.Tensor] | None = None,
    ) -> tuple[str, dict]:
        prompt = self._render_prompt(query, passages)
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        t0 = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.3,
        )
        generation_time = time.perf_counter() - t0
        new_tokens = output_ids[0, prompt_tokens:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        stats = {
            "generated_tokens": int(new_tokens.shape[-1]),
            "generation_time_s": generation_time,
        }
        return answer, stats


# ---------------------------------------------------------------------------
# LatentGenerator  (encoder-decoder, latent-space injection into decoder)
# ---------------------------------------------------------------------------

class LatentGenerator:
    def __init__(
        self,
        generator_model: str = DEFAULT_LATENT_GENERATOR,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.generator_model = generator_model
        self.max_new_tokens = max_new_tokens
        print(f"Loading latent generator model (Seq2Seq): {generator_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.model = _load_to_device(
            AutoModelForSeq2SeqLM.from_pretrained(
                generator_model,
                torch_dtype=_generator_dtype(),
            ),
            "latent generator",
        ).eval()

    def _encode_passage(self, passage: Passage) -> torch.Tensor:
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            passage.text,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        latents = self.model.get_encoder()(**enc).last_hidden_state[0]
        seq_len = enc["attention_mask"][0].sum().item()
        return latents[:seq_len, :]

    @torch.no_grad()
    def generate(
        self,
        query: str,
        passages: list[Passage],
        passage_latents: list[torch.Tensor] | None = None,
    ) -> tuple[str, dict]:
        device = next(self.model.parameters()).device
        q_enc = self.tokenizer(
            f"Answer the question using the context: {query}",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        q_latents = self.model.get_encoder()(**q_enc).last_hidden_state[0]
        if passage_latents is None:
            passage_latents = [self._encode_passage(p) for p in passages]
        batch_latents = [q_latents] + passage_latents
        combined_latents = torch.cat(batch_latents, dim=0).unsqueeze(0)
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=combined_latents)
        start_token_id = (
            getattr(self.model.generation_config, "decoder_start_token_id", None)
            or getattr(self.model.config, "decoder_start_token_id", None)
            or self.tokenizer.pad_token_id
        )
        t0 = time.perf_counter()
        output_ids = self.model.generate(
            encoder_outputs=encoder_outputs_obj,
            decoder_input_ids=torch.tensor([[start_token_id]], device=device),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generation_time = time.perf_counter() - t0
        answer = self.tokenizer.decode(output_ids[0, 1:], skip_special_tokens=True).strip()
        stats = {
            "generated_tokens": int(output_ids.shape[-1]) - 1,
            "generation_time_s": generation_time,
        }
        return answer, stats


# ---------------------------------------------------------------------------
# RAGPipeline  (generic orchestrator)
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(
        self,
        retriever: BGERTRetriever | LatentRetriever,
        generator: TextGenerator | LatentGenerator,
        top_k: int = DEFAULT_TOP_K,
        seed: int = SEED,
    ):
        set_seed(seed)
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> Result:
        t0 = time.perf_counter()
        passages, passage_latents, retrieval_time = self.retriever.retrieve(query, self.top_k)
        answer, stats = self.generator.generate(query, passages, passage_latents)
        return Result(
            query=query,
            answer=answer,
            retrieved_passage_ids=[p.passage_id for p in passages],
            retrieved_source_doc_ids=[p.source_doc_id for p in passages],
            retrieval_time_s=retrieval_time,
            generation_time_s=stats["generation_time_s"],
            total_time_s=time.perf_counter() - t0,
            generated_tokens=stats["generated_tokens"],
        )
