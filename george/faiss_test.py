import numpy as np
import faiss


def run():
    # ---- config ----
    dim = 768        # embedding size (e.g. T5 hidden dim)
    num_docs = 1000  # number of stored vectors
    k = 5            # top-k retrieval

    # ---- create dummy embeddings (replace with your encoder outputs) ----
    np.random.seed(42)
    xb = np.random.random((num_docs, dim)).astype("float32")

    # normalize if you want cosine similarity
    faiss.normalize_L2(xb)

    # ---- build index (Inner Product = cosine after normalization) ----
    index = faiss.IndexFlatIP(dim)

    # ---- add vectors to index ----
    index.add(xb)

    print(f"Indexed {index.ntotal} vectors")

    # ---- query ----
    xq = np.random.random((1, dim)).astype("float32")
    faiss.normalize_L2(xq)

    distances, indices = index.search(xq, k)

    print("Top-k indices:", indices)
    print("Similarity scores:", distances)


if __name__ == "__main__":
    run()

