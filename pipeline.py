"""
pipeline.py - Fully decoupled Latent RAG pipeline using T5Encoder/Seq2Seq.

Offline:
    passage JSONL -> T5 Encoder -> mean-pooled FAISS index + safetensors 2D sequence latents

Online:
    query -> FAISS search -> fetch top-k safetensors matrix latents -> bypass encoder -> manual generation loop
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_EMBEDDING_MODEL = "google/t5gemma-2-270m-270m"
DEFAULT_GENERATOR_MODEL = "google/t5gemma-2-270m-270m"
DEFAULT_TOP_K = 5
DEFAULT_MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

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
        print(f"Loading embedding model (Encoder only): {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        dtype = _generator_dtype() if DEVICE == "cuda" else torch.float32
        
        # Load Seq2Seq and extract encoder to save VRAM
        full_model = AutoModelForSeq2SeqLM.from_pretrained(embedding_model, torch_dtype=dtype)
        self.model = _load_to_device(full_model.get_encoder(), "embedding encoder").eval()
        
        self.index: faiss.IndexFlatIP | None = None
        self.passages: list[Passage] = []
        self.all_latents: dict[str, torch.Tensor] = {}
        self.latent_file: str | None = None

    @torch.no_grad()
    def _embed(self, texts: list[str]) -> tuple[np.ndarray, list[torch.Tensor]]:
        # T5-style prompt or raw text
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
            # Strip padding tokens out of the sequence representation
            seq_len = attention_mask[i].sum().item()
            true_latents = latents[i, :seq_len, :]
            saved_latents.append(true_latents)
            
            # Mean pool for FAISS search compatibility
            mean_pooled = true_latents.mean(dim=0).unsqueeze(0)
            faiss_vec = F.normalize(mean_pooled, p=2, dim=-1).float().numpy()
            faiss_vecs.append(faiss_vec)

        return np.concatenate(faiss_vecs, axis=0), saved_latents

    def build_index(self, passages: list[Passage], batch_size: int = 256) -> None:
        if not passages:
            raise ValueError("Cannot build an index with zero passages.")

        self.passages = passages
        n_batches = (len(passages) + batch_size - 1) // batch_size
        print(f"Embedding {len(passages)} passages ({n_batches} batches)...")
        
        all_embeddings = []
        self.all_latents = {}
        
        for i, start in enumerate(range(0, len(passages), batch_size)):
            batch = passages[start : start + batch_size]
            faiss_batch, latents_batch = self._embed([passage.text for passage in batch])
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

        # Save massive latents via safetensors
        print("Saving Latent Sequences to Safetensors...")
        save_file(self.all_latents, str(index_dir / "latents.safetensors"))

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
                "Index and metadata size mismatch: "
                f"index has {self.index.ntotal} rows but metadata has {len(self.passages)} passages."
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
        
        # Lazy load sequence matrices directly into VRAM to avoid OOM
        latents_list = []
        with safe_open(self.latent_file, framework="pt", device="cpu") as f:
            for p in passages:
                lat = f.get_tensor(p.passage_id).to(device)
                latents_list.append(lat)
                
        elapsed = time.perf_counter() - t0
        return passages, latents_list, elapsed


class Generator:
    def __init__(
        self,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.generator_model = generator_model
        self.max_new_tokens = max_new_tokens

        print(f"Loading generator model (Seq2Seq Full): {generator_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        
        self.model = _load_to_device(
            AutoModelForSeq2SeqLM.from_pretrained(
                generator_model,
                torch_dtype=_generator_dtype(),
            ),
            "generator model",
        ).eval()

    @torch.no_grad()
    def generate(self, query: str, passage_latents: list[torch.Tensor]) -> tuple[str, dict]:
        device = next(self.model.parameters()).device
        
        # Encode the query live
        q_enc = self.tokenizer(f"Question: {query}", return_tensors="pt").to(device)
        q_latents = self.model.get_encoder()(**q_enc).last_hidden_state[0] # [q_len, hidden]
        
        # Combine Latent Space Matrices (1 query sequence + k passage sequences)
        batch_latents = [q_latents] + passage_latents
        combined_latents = torch.cat(batch_latents, dim=0).unsqueeze(0) # [1, total_len, hidden_dim]

        # Manual Greedy Decoding Loop injection via encoder_outputs
        start_token_id = getattr(self.model.config, "decoder_start_token_id", None)
        if start_token_id is None:
            start_token_id = getattr(self.model.config, "bos_token_id", self.tokenizer.pad_token_id)
            
        decoder_input_ids = torch.tensor([[start_token_id]], device=device)
        
        # Wrap the latents in a BaseModelOutput object for the new architectures!
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=combined_latents)
        
        t0 = time.perf_counter()
        generated_tokens = 0
        
        for _ in range(self.max_new_tokens):
            outputs = self.model(
                encoder_outputs=encoder_outputs_obj,
                decoder_input_ids=decoder_input_ids,
            )
            # Pick highest prob token
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
        
        # New Unpacking includes memory-heavy sequence tensors!
        retrieved_passages, passage_latents, retrieval_time = self.retriever.retrieve(query, self.top_k)
        
        # Generator bypasses texts and takes passage_latents directly
        answer, stats = self.generator.generate(query, passage_latents)

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
