"""
pipeline.py - Flexible RAG pipeline supporting traditional and latent-space modes.

Retrievers:
    BGERTRetriever   - BGE dense retrieval (AutoModel + [CLS] pooling)
    LatentRetriever  - T5 encoder latent retrieval (safetensors + MaxSim)

Generators:
    TextGenerator    - causal LM text generation from prompted passages
    LatentGenerator  - encoder-decoder generation from latent representations

Pipeline:
    RAGPipeline      - generic orchestrator accepting any retriever + generator
"""
from __future__ import annotations

import gc
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
DEFAULT_LATENT_MODEL = "google/t5gemma-2b-2b-ul2"
DEFAULT_TEXT_GENERATOR = "Qwen/Qwen3.5-0.8B"
DEFAULT_LATENT_GENERATOR = "google/t5gemma-2b-2b-ul2"
DEFAULT_EMBEDDING_MODEL = DEFAULT_LATENT_MODEL
DEFAULT_GENERATOR_MODEL = DEFAULT_LATENT_GENERATOR
DEFAULT_TOP_K = 5
DEFAULT_MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

PROMPT_PREFIX = "Context:\n"
PROMPT_SUFFIX_TEMPLATE = "\n\nQuestion: {question}\n\nShort answer (1-5 words):"


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
    retriever_type: str = "latent"
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


class LatentRetriever:
    def __init__(self, embedding_model: str = DEFAULT_LATENT_MODEL):
        self.embedding_model = embedding_model
        print(f"Loading latent embedding model (encoder only): {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        dtype = _generator_dtype() if DEVICE == "cuda" else torch.float32
        full_model = AutoModelForSeq2SeqLM.from_pretrained(embedding_model, torch_dtype=dtype)
        self.model = _load_to_device(full_model.get_encoder(), "latent encoder").eval()
        self.passages: list[Passage] = []
        self.all_latents: dict[str, torch.Tensor] = {}
        self.raw_latents_cpu: list[torch.Tensor] = []

    @torch.no_grad()
    def _embed(self, texts: list[str], is_query: bool = False) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device
        prefix = "Query: " if is_query else "Document: "
        encoded = self.tokenizer(
            [prefix + text for text in texts],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**encoded)
        latents = outputs.last_hidden_state.cpu()
        attention_mask = encoded["attention_mask"].cpu().bool()

        saved_latents = []
        for i in range(len(texts)):
            saved_latents.append(latents[i][attention_mask[i]].clone().contiguous())
        return saved_latents

    def build_index(
        self,
        passages: list[Passage],
        batch_size: int = 64,
        index_dir: str | Path | None = None,
    ) -> None:
        if not passages:
            raise ValueError("Cannot build an index with zero passages.")
        self.passages = passages
        n_batches = (len(passages) + batch_size - 1) // batch_size
        print(f"Embedding {len(passages)} passages ({n_batches} batches)...")

        self.all_latents = {}
        shard_idx = 0
        shard_size = 2000
        output_dir = Path(index_dir) if index_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, start in enumerate(range(0, len(passages), batch_size)):
            batch = passages[start : start + batch_size]
            latents_batch = self._embed([passage.text for passage in batch], is_query=False)
            for passage, latents in zip(batch, latents_batch):
                self.all_latents[passage.passage_id] = latents

            if output_dir is not None and len(self.all_latents) >= shard_size:
                save_file(self.all_latents, str(output_dir / f"latents_{shard_idx}.safetensors"))
                self.all_latents.clear()
                shard_idx += 1
                gc.collect()

            if (i + 1) % 10 == 0 or i + 1 == n_batches:
                print(f"  batch {i + 1}/{n_batches}")

        if output_dir is not None and self.all_latents:
            save_file(self.all_latents, str(output_dir / f"latents_{shard_idx}.safetensors"))
            self.all_latents.clear()
            gc.collect()

    def save(
        self,
        index_dir: str | Path,
        corpus_path: str,
        max_docs: int | None = None,
        doc_id_provided_count: int | None = None,
        doc_id_missing_count: int | None = None,
    ) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for passage in self.passages:
                handle.write(json.dumps(asdict(passage), ensure_ascii=True) + "\n")

        if self.all_latents:
            print("Saving latent sequences to safetensors...")
            save_file(self.all_latents, str(index_dir / "latents_0.safetensors"))
            self.all_latents.clear()
            gc.collect()

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
        metadata_path = index_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.passages = []
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.passages.append(Passage(**record))

        config = load_index_config(index_dir)
        print("Loading sequence latents into RAM for exhaustive MaxSim search...")
        shard_files = sorted(index_dir.glob("latents_*.safetensors"))
        if not shard_files:
            legacy_file = index_dir / "latents.safetensors"
            if legacy_file.exists():
                shard_files = [legacy_file]
            else:
                raise FileNotFoundError(f"Safetensors latents not found in {index_dir}")

        latents_by_id: dict[str, torch.Tensor] = {}
        for shard_file in shard_files:
            with safe_open(str(shard_file), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    latents_by_id[key] = handle.get_tensor(key)

        self.raw_latents_cpu = []
        for passage in self.passages:
            self.raw_latents_cpu.append(latents_by_id[passage.passage_id])

        if len(self.raw_latents_cpu) != len(self.passages):
            raise ValueError(
                "Metadata size mismatch: "
                f"latents file has {len(self.raw_latents_cpu)} rows but metadata has {len(self.passages)} passages."
            )
        return config

    def retrieve(self, query: str, top_k: int) -> tuple[list[Passage], list[torch.Tensor], float]:
        if not self.passages or not self.raw_latents_cpu:
            raise ValueError("Retriever index is not loaded.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        t0 = time.perf_counter()
        device = next(self.model.parameters()).device
        query_latents = self._embed([query], is_query=True)[0].to(device).float()
        q_norm = F.normalize(query_latents, p=2, dim=-1)

        chunk_size = 1000
        all_scores = []
        for i in range(0, len(self.raw_latents_cpu), chunk_size):
            chunk = self.raw_latents_cpu[i : i + chunk_size]
            padded_chunk = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True).to(device).float()
            padded_norm = F.normalize(padded_chunk, p=2, dim=-1)
            lengths = torch.tensor([len(item) for item in chunk], device=device)
            max_len = padded_chunk.shape[1]
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            sim = torch.einsum("qd,npd->nqp", q_norm, padded_norm)
            sim = sim.masked_fill(~mask.unsqueeze(1), -1e9)
            max_sim, _ = sim.max(dim=2)
            all_scores.append(max_sim.sum(dim=1).cpu())

        scores = torch.cat(all_scores, dim=0)
        _, top_indices = torch.topk(scores, min(top_k, len(self.passages)))
        top_indices_list = top_indices.tolist()
        passages = [self.passages[idx] for idx in top_indices_list]
        latents_list = [self.raw_latents_cpu[idx].to(device) for idx in top_indices_list]
        elapsed = time.perf_counter() - t0
        return passages, latents_list, elapsed


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
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Short answer (1-5 words):"
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
            f"Question: {query}\n\nShort answer:",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        q_latents = self.model.get_encoder()(**q_enc).last_hidden_state[0]
        if passage_latents is None:
            passage_latents = [self._encode_passage(p) for p in passages]
        combined_latents = torch.cat([q_latents] + passage_latents, dim=0).unsqueeze(0)
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=combined_latents)
        start_token_id = (
            getattr(self.model.generation_config, "decoder_start_token_id", None)
            or getattr(self.model.config, "decoder_start_token_id", None)
            or getattr(self.model.config, "bos_token_id", None)
            or self.tokenizer.pad_token_id
        )
        decoder_input_ids = torch.tensor([[start_token_id]], device=device)

        t0 = time.perf_counter()
        generated_tokens = 0
        for _ in range(self.max_new_tokens):
            outputs = self.model(
                encoder_outputs=encoder_outputs_obj,
                decoder_input_ids=decoder_input_ids,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            generated_tokens += 1
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generation_time = time.perf_counter() - t0
        answer = self.tokenizer.decode(decoder_input_ids[0, 1:], skip_special_tokens=True).strip()
        stats = {
            "generated_tokens": generated_tokens,
            "generation_time_s": generation_time,
        }
        return answer, stats


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
