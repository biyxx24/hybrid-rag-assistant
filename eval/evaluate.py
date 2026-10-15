"""Compare retrieval strategies on a small gold set of questions.

Each gold question names the document(s) that hold its answer. A strategy scores a *hit* when at
least one of its top-k chunks comes from such a document, and its *MRR* (mean reciprocal rank)
rewards finding that document near the top. The gold set lives in ``eval/gold_questions.json``.

Usage (from the project root):
    python -m eval.evaluate            # real models, needs an API key
    python -m eval.evaluate --fake     # offline smoke test of the harness itself
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

STRATEGIES = {
    "baseline (question only)": {"rewrites": 0, "use_hyde": False},
    "multi-query (3 rewrites + RRF)": {"rewrites": 3, "use_hyde": False},
    "HyDE": {"rewrites": 0, "use_hyde": True},
    "multi-query + HyDE": {"rewrites": 3, "use_hyde": True},
}


def evaluate(pipeline, gold: list[dict], k: int) -> list[dict]:
    from hybrid_rag.pipeline import Options

    rows = []
    for name, params in STRATEGIES.items():
        options = Options(k=k, **params)
        hits = 0
        reciprocal_ranks = 0.0
        calls_before = pipeline.llm.usage.calls
        for item in gold:
            _, _, retrieved = pipeline.retrieve_docs(item["question"], options)
            expected = set(item["expected_sources"])
            first_rank = next((hit.rank for hit in retrieved if hit.source in expected), None)
            if first_rank is not None:
                hits += 1
                reciprocal_ranks += 1.0 / first_rank
        rows.append(
            {
                "strategy": name,
                f"hit@{k}": hits / len(gold),
                "MRR": reciprocal_ranks / len(gold),
                "model calls / question": (pipeline.llm.usage.calls - calls_before) / len(gold),
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    for row in rows:
        cells = [f"{value:.2f}" if isinstance(value, float) else str(value) for value in row.values()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies on the gold set")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--fake", action="store_true", help="offline mode (hash embeddings, fake model)")
    parser.add_argument("--gold", default=str(Path(__file__).with_name("gold_questions.json")))
    parser.add_argument("--out", default=str(Path(__file__).with_name("results.md")))
    args = parser.parse_args()
    if args.fake:
        os.environ["RAG_FAKE"] = "1"

    from hybrid_rag.pipeline import Pipeline  # imported after RAG_FAKE is set

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    pipeline = Pipeline()
    rows = evaluate(pipeline, gold, args.k)
    table = to_markdown(rows)

    mode = "offline fake models" if pipeline.settings.fake else (
        f"chat `{pipeline.settings.chat_model}`, embeddings `{pipeline.settings.embed_model}`"
    )
    report = (
        f"# Retrieval evaluation\n\n{time.strftime('%Y-%m-%d %H:%M')} · {len(gold)} questions · "
        f"k={args.k} · {mode}\n\n{table}\n"
    )
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
