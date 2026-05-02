import os
import json
from typing import Dict, List

import faiss
import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutput


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
STATES_DIR: str = os.path.join(DATA_DIR, "states")
INDEX_PATH: str = os.path.join(DATA_DIR, "index.faiss")
METADATA_PATH: str = os.path.join(DATA_DIR, "metadata.json")


def save_hidden_states(path: str, hidden_states: Tensor, attention_mask: Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "hidden_states": hidden_states.cpu(),   # [1, seq_len, hidden_dim]
            "attention_mask": attention_mask.cpu(), # [1, seq_len]
        },
        path,
    )


def mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """
    last_hidden_state: [1, seq_len, hidden_dim]
    attention_mask:   [1, seq_len]
    returns:          [hidden_dim]
    """
    hidden_states: Tensor = last_hidden_state[0]   # [seq_len, hidden_dim]
    mask: Tensor = attention_mask[0].unsqueeze(-1).to(hidden_states.dtype)  # [seq_len, 1]

    masked_hidden: Tensor = hidden_states * mask
    pooled: Tensor = masked_hidden.sum(dim=0) / mask.sum().clamp(min=1e-9)
    return pooled


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 8,
    overlap: int = 2,
) -> List[str]:
    """
    Splits text into token-based chunks and decodes them back into strings.

    max_tokens: number of tokens per chunk
    overlap:    number of overlapping tokens between neighboring chunks
    """
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    input_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)

    if not input_ids:
        return []

    chunks: List[str] = []
    step: int = max_tokens - overlap

    for start in range(0, len(input_ids), step):
        end = start + max_tokens
        chunk_ids: List[int] = input_ids[start:end]

        if not chunk_ids:
            continue

        chunk_text_str: str = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text_str:
            chunks.append(chunk_text_str)

        if end >= len(input_ids):
            break

    return chunks


def main() -> None:
    model_name: str = "google/t5gemma-2-4b-4b"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Example: each entry here is a full document, not a chunk
    documents: List[Dict[str, str]] = [
        {
            "doc_id": "doc-12",
            "text": "Leaves are usually green. Healthy plants use chlorophyll for photosynthesis. "
                    "In autumn, leaves can also turn yellow, orange, or red.",
        },
        {
            "doc_id": "doc-13",
            "text": "The sky is usually blue during the day because of Rayleigh scattering. "
                    "At sunrise and sunset, it may look orange or red.",
        },
        {
            "doc_id": "doc-14",
            "text": "Fire trucks are usually red, although some countries or cities use other colors "
                    "such as yellow or lime for visibility.",
        },
    ]

    os.makedirs(STATES_DIR, exist_ok=True)

    pooled_vectors: List[Tensor] = []
    metadata: Dict[str, Dict[str, str]] = {}

    for document in documents:
        doc_id: str = document["doc_id"]
        doc_text: str = document["text"]

        chunks: List[str] = chunk_text(
            text=doc_text,
            tokenizer=tokenizer,
            max_tokens=8,
            overlap=2,
        )

        print(f"{doc_id}: {len(chunks)} chunk(s)")

        for chunk_idx, chunk_text_str in enumerate(chunks):
            chunk_id: str = f"{doc_id}:{chunk_idx}"

            inputs: Dict[str, Tensor] = tokenizer(chunk_text_str, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                encoder_outputs: BaseModelOutput = model.get_encoder()(**inputs)

            print(f"  {chunk_id} shape: {encoder_outputs.last_hidden_state.shape}")

            state_path: str = os.path.join(STATES_DIR, f"{chunk_id.replace(':', '_')}.pt")

            save_hidden_states(
                path=state_path,
                hidden_states=encoder_outputs.last_hidden_state,
                attention_mask=inputs["attention_mask"],
            )

            pooled: Tensor = mean_pool(
                last_hidden_state=encoder_outputs.last_hidden_state,
                attention_mask=inputs["attention_mask"],
            )

            pooled_vectors.append(pooled.to(torch.float32).cpu())

            metadata[str(len(pooled_vectors) - 1)] = {
                "id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text_str,
                "state_path": state_path,
            }

    if not pooled_vectors:
        raise ValueError("No chunks were created. Check your documents or chunking logic.")

    matrix = torch.stack(pooled_vectors).cpu().numpy().astype("float32")
    # shape: [num_chunks, hidden_dim]

    hidden_dim: int = matrix.shape[1]
    index = faiss.IndexFlatL2(hidden_dim)
    index.add(matrix)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(pooled_vectors)} chunks to FAISS.")
    print(f"FAISS index: {INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Hidden states: {STATES_DIR}")


if __name__ == "__main__":
    main()