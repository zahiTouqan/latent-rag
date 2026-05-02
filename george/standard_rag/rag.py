import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from typing import Dict, List

import faiss
import torch
from torch import Tensor
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
INDEX_PATH: str = os.path.join(DATA_DIR, "index.faiss")
METADATA_PATH: str = os.path.join(DATA_DIR, "metadata.json")


def mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """
    last_hidden_state: [1, seq_len, hidden_dim]
    attention_mask:   [1, seq_len]
    returns:          [hidden_dim]
    """
    hidden_states: Tensor = last_hidden_state[0]
    mask: Tensor = attention_mask[0].unsqueeze(-1).to(hidden_states.dtype)
    masked_hidden: Tensor = hidden_states * mask
    pooled: Tensor = masked_hidden.sum(dim=0) / mask.sum().clamp(min=1e-9)
    return pooled


def load_metadata(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top_k_chunks(
    query: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    index: faiss.Index,
    metadata: Dict[str, Dict[str, str]],
    top_k: int,
) -> List[Dict[str, str]]:
    query_inputs: Dict[str, Tensor] = tokenizer(query, return_tensors="pt")
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}

    with torch.no_grad():
        query_encoder_outputs: BaseModelOutput = model.get_encoder()(**query_inputs)

    query_vector: Tensor = mean_pool(
        query_encoder_outputs.last_hidden_state,
        query_inputs["attention_mask"],
    )

    query_vector_np = query_vector.to(torch.float32).cpu().numpy().reshape(1, -1)

    k: int = min(top_k, index.ntotal)
    distances, indices = index.search(query_vector_np, k=k)

    retrieved_chunks: List[Dict[str, str]] = []

    print(f"Query: {query}")
    print(f"\nTop {k} chunks by L2 distance:")

    for rank, (chunk_idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        row_idx: int = int(chunk_idx)
        chunk_info: Dict[str, str] = metadata[str(row_idx)]

        print(f"{rank}. FAISS row: {row_idx}")
        print(f"   Chunk id: {chunk_info['id']}")
        print(f"   Parent doc_id: {chunk_info['doc_id']}")
        print(f"   Text: {chunk_info['text']}")
        print(f"   L2 distance: {float(dist):.4f}")

        retrieved_chunks.append(chunk_info)

    return retrieved_chunks


def build_context_text(chunks: List[Dict[str, str]]) -> str:
    return "\n".join(chunk["text"] for chunk in chunks)


def main() -> None:
    model_name: str = "google/t5gemma-2-4b-4b"
    top_k: int = 3

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
        device
    )

    index = faiss.read_index(INDEX_PATH)
    metadata = load_metadata(METADATA_PATH)

    query: str = "What color is a leaf?"

    top_chunks: List[Dict[str, str]] = retrieve_top_k_chunks(
        query=query,
        tokenizer=tokenizer,
        model=model,
        device=device,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    context_text: str = build_context_text(top_chunks)

    print("\nCombined text context:")
    print(context_text)

    generation_prompt: str = (
        "Answer the question using the context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    model_inputs: Dict[str, Tensor] = tokenizer(
        generation_prompt,
        return_tensors="pt",
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=20,
        )

    result: str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nRetrieved chunks used for generation:")
    for i, chunk in enumerate(top_chunks, start=1):
        print(f"{i}. {chunk['id']} | {chunk['text']}")

    print("\nGenerated answer:")
    print(result)


if __name__ == "__main__":
    main()
