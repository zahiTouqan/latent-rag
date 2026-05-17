from __future__ import annotations

import json
from pathlib import Path


COLUMNS = [
    "em",
    "f1",
    "recall@5",
    "answer_support@5",
    "visible_special_tokens",
    "max_repeated_3gram",
    "latency_p50_ms",
    "latency_p95_ms",
]


def main() -> None:
    files = sorted(Path("results").glob("results_*.json"), key=lambda path: path.stat().st_mtime)
    if not files:
        print("No result files found in results/")
        return

    header = f"{'File':<42s} {'Mode':<12s}" + "".join(f"{column:>24s}" for column in COLUMNS)
    print(header)
    print("-" * len(header))
    for result_file in files:
        with result_file.open(encoding="utf-8") as handle:
            data = json.load(handle)
        mode = data.get("config", {}).get("mode", "N/A")
        metrics = data.get("metrics", {})
        values = "".join(f"{float(metrics.get(column, 0.0)):>24.4f}" for column in COLUMNS)
        print(f"{result_file.name:<42s} {mode:<12s}{values}")


if __name__ == "__main__":
    main()
