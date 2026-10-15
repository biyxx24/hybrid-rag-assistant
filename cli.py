"""Ask one question from the terminal.

Examples:
    python cli.py "How long do I have to request a refund?"
    python cli.py "Which genre sells the most tracks?" --trace
    RAG_FAKE=1 python cli.py "Hello" --route chat
"""

from __future__ import annotations

import argparse
import json

from hybrid_rag.pipeline import Options, Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG Assistant command line")
    parser.add_argument("question")
    parser.add_argument("--route", default="auto", choices=["auto", "docs", "sql", "both", "chat"])
    parser.add_argument("-k", type=int, default=4, help="chunks kept after fusion")
    parser.add_argument("--rewrites", type=int, default=3, help="alternative phrasings to generate")
    parser.add_argument("--no-hyde", action="store_true", help="disable HyDE")
    parser.add_argument("--trace", action="store_true", help="print the pipeline trace as JSON")
    args = parser.parse_args()

    pipeline = Pipeline()
    options = Options(route=args.route, k=args.k, rewrites=args.rewrites, use_hyde=not args.no_hyde)
    answer, trace = pipeline.answer(args.question, options=options)

    print(answer)
    if args.trace:
        print("\n--- trace ---")
        print(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
