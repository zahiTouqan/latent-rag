import json
import os

files = [
    "results_baseline.json",
    "results_t5gemma_embed.json",
    "results_T5Gemma_2B_embedings.json"
]

print("| File | Generator | Embedding | EM | F1 | Recall@5 | Total Time |")
print("|---|---|---|---|---|---|---|")

for f in files:
    if os.path.exists(f):
        with open(f, 'r') as fp:
            data = json.load(fp)
            c = data.get("config", {})
            m = data.get("metrics", {})
            gen = c.get("generator_model", "N/A").split('/')[-1]
            idx = c.get("index", {})
            if isinstance(idx, dict):
                emb = idx.get("embedding_model", "N/A").split('/')[-1]
            else:
                emb = "N/A"
            em = m.get("em", 0)
            f1 = m.get("f1", 0)
            rec = m.get("recall@5", 0)
            time = m.get("total_time_s", 0)
            print(f"| {f} | {gen} | {emb} | {em:.4f} | {f1:.4f} | {rec:.4f} | {time:.2f}s |")
