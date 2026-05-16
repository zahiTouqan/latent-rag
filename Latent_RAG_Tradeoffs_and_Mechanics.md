# Latent RAG: Trade-offs and Mechanics

This document outlines the core concepts, trade-offs, and underlying mechanics of the Latent RAG (Dense Memory Injection) architecture, specifically focusing on injecting continuous passage embeddings directly into the decoder of an Encoder-Decoder model.

## 1. The Core Trade-offs

Traditional RAG retrieves raw text passages, concatenates them with the query (`Query: {q} Context: {c1, c2...}`), and forces the model to process all of that text from scratch. Latent RAG bypasses the text bottleneck by directly injecting pre-computed tensor matrices (the latents) into the decoder.

### 🟢 Advantages

**1. Massive Reduction in Online Compute (Latency)**
In traditional RAG, the generative model has to re-encode the retrieved passages every single time a user asks a question. If you retrieve 5 long passages, the encoder computes self-attention over thousands of tokens dynamically. 
By pre-computing the passage embeddings offline, the online generator only needs to encode the short user query. It then stitches the pre-computed matrices together. This heavily reduces the Floating Point Operations (FLOPs) required at runtime, leading to faster Time-To-First-Token (TTFT).

**2. Bypassing Strict Token Limits**
Language models have fixed context windows (e.g., 512 or 2048 tokens). In traditional RAG, if you exceed this limit, you must truncate the text. By operating directly in the latent space, you are restricted primarily by GPU VRAM, not the tokenizer's sequence limit. You can theoretically inject dozens of passage matrices directly into the decoder's cross-attention layer.

**3. Native Alignment**
Text is lossy. When you convert a concept into a discrete string of text, you lose some semantic nuance. By retrieving and passing the dense, high-dimensional latent representations directly, the decoder gets to "see" the exact continuous features the encoder extracted, which can theoretically provide richer semantic context.

### 🔴 Disadvantages

**1. Massive Storage and RAM Overhead**
Storing raw text is incredibly cheap. Storing token-level hidden states is extremely expensive. A single passage might be 500 characters of text (a few kilobytes), but its latent representation in float16 (e.g., a `[128 tokens, 1024 hidden_dim]` matrix) takes up significantly more disk space. Managing multi-gigabyte matrices requires efficient storage formats (like `.safetensors`) and loading them into RAM requires a lot of overhead compared to simple text retrieval.

**2. Loss of "Joint Encoding" (The Cross-Attention Disconnect)**
This is arguably the most critical drawback for generation quality:
* **In Traditional RAG:** The encoder processes the Query and the Context *together*. The self-attention mechanism allows the passage tokens to attend directly to the query tokens. The model highlights the relevant parts of the passage based on what the query is asking.
* **In Latent RAG:** The passages are encoded completely in isolation (offline). The encoder has no idea what the future query will be. The query and the passages never interact until the very end of the pipeline inside the decoder. This "late fusion" means the passage representations aren't dynamically tailored to the question, which often causes a drop in F1 scores and Exact Match metrics compared to a model that reads the text dynamically.

**3. Memory Bandwidth Bottlenecks**
While compute (FLOPs) is saved by not re-encoding text, memory bandwidth is heavily taxed. Moving gigabytes of dense tensor matrices from system RAM to GPU VRAM for every single query can become a severe bottleneck, potentially negating the compute speedups if not engineered perfectly.

**4. Architectural Constraints**
This pipeline relies on **Encoder-Decoder** models (T5, BART), which naturally separate the encoder (which generates the latents) and the decoder (which attends to them). Modern **Decoder-Only** models (like GPT-4, Llama 3, Qwen) lack this separation, making it incredibly difficult to inject raw latents without training custom projection layers or soft-prompt adapters.

---

## 2. Mechanics of Embedding Injection

In an Encoder-Decoder architecture like T5, the retrieved passage embeddings are passed to the decoder using the **Cross-Attention** mechanism.

### The Code (How it is injected)
In the generation pipeline, the encoder is bypassed, and the decoder is manually handed the latents using the `encoder_outputs` argument:

```python
# 1. Encode query
q_enc = tokenizer(f"Question: {query}", return_tensors="pt")
q_latents = model.get_encoder()(**q_enc).last_hidden_state[0]

# 2. Combine with retrieved passage latents
batch_latents = [q_latents] + passage_latents
combined_latents = torch.cat(batch_latents, dim=0).unsqueeze(0)

# 3. Inject directly into Decoder
encoder_outputs_obj = BaseModelOutput(last_hidden_state=combined_latents)

outputs = model(
    encoder_outputs=encoder_outputs_obj, # <--- The injection point
    decoder_input_ids=decoder_input_ids,
)
```
When `encoder_outputs` is passed to the model, the tensor matrix is routed straight into the decoder's Cross-Attention layers.

### The Math (How the Decoder uses it)
Once inside the decoder, the model uses standard Transformer **Attention**: 
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

In **Cross-Attention**, the matrices are split up as follows:
* **The Query (Q):** This comes from the **Decoder's** current state. It represents what the decoder is currently "thinking" about as it tries to predict the next word.
* **The Keys (K):** This comes directly from the **RAG Embeddings** (`combined_latents`). It acts as a searchable index of the retrieved passages.
* **The Values (V):** This also comes directly from the **RAG Embeddings**. It holds the actual semantic content of the passages.

### The Step-by-Step Multiplication
As the decoder generates the answer word-by-word, the following happens at every cross-attention layer step:

1. **Calculate Relevance (Q × K):** The decoder takes its current state (Q) and performs a dot-product multiplication against all the Keys (K) from the retrieved passages. This calculates an array of "attention scores," determining which specific token embeddings in the retrieved passages are most relevant to the word it needs to generate.
2. **Apply Softmax:** The scores are normalized so they add up to 100% (1.0). 
3. **Extract Information (Scores × V):** The model multiplies those normalized percentages by the Values (V) (the retrieved embeddings). If a specific token in a passage receives a 90% attention score, 90% of its embedding vector is pulled directly into the decoder.
4. **Generate Token:** The decoder uses that pulled information to predict the next word in the sequence.

Instead of appending the embeddings to an output, the retrieved embeddings act as a **static, continuous database (Keys and Values)**. As the decoder generates text dynamically, it repeatedly queries this database via dot-product multiplication to pull the exact semantic features it needs to answer the question.

---

## 3. The Pipeline Shift: Where Do the Embeddings Actually Enter?

A major conceptual hurdle in Latent RAG is understanding that the text passages literally disappear from the pipeline during generation. Here is a side-by-side comparison of where the data enters:

### Standard Text-Based RAG (The Normal Way)
In a standard RAG system, the generator always deals with **Text strings**.
1. **Search:** The system retrieves text strings (e.g., `"Reba McEntire sang Does He Love You."`)
2. **Concatenation:** The retrieved text is pasted next to the user's query into one massive string:
   `Text = "Context: Reba McEntire sang... Question: Who sings with Reba?"`
3. **The Encoder:** The model's Encoder reads this massive string of text and converts the whole thing into embeddings *dynamically*.
4. **The Decoder:** The Decoder then generates the answer.

### Latent RAG (Dense Memory Injection)
In Latent RAG, the retrieved passages **never exist as text during generation**. They enter the system directly as **Tensor Matrices**.

**Step 1: The Offline Pre-computation (Where the text used to be)**
During the index building phase, the text passages are run through the Encoder ahead of time. The text is converted into mathematical matrices (the latent embeddings) and saved to a database (e.g., a `.safetensors` file). *At this exact moment, the text strings are discarded.*

**Step 2: The Online Generation (Where the embeddings enter)**
During inference, when a user asks a question:
1. **Search:** The retriever searches the index and pulls out the **pre-computed matrices (the embeddings)**, not text.
2. **Bypassing the Encoder:** These matrices are *not* passed through the model's Encoder, because they are already encoded. 
3. **The Injection Point:** The system encodes *only* the user's short query text, and then stitches the query embedding together with the retrieved passage matrices. 
4. **The Decoder:** These combined matrices are handed straight to the **Decoder** via the `encoder_outputs` argument.

**In summary:** In standard RAG, the passages enter the pipeline at the very beginning as **Text Strings** fed into the **Encoder**. In Latent RAG, the passages enter the pipeline right at the very end as **Tensor Matrices** injected directly into the **Decoder**.
