"""
Run the DocInsight evaluation.

Examples:
    # Offline lexical baseline, no API key needed
    python -m evaluation.run --retrievers bm25

    # The app's real pipeline plus hybrid and answer faithfulness (needs OPENAI_API_KEY)
    python -m evaluation.run --retrievers bm25,dense,hybrid --faithfulness 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from evaluation.chunking import chunk_documents, label_relevant_chunks
from evaluation.dataset import load_squad
from evaluation.metrics import answer_recall, mean, reciprocal_rank, recall_at_k, token_f1
from evaluation.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from utils.config import AppConfig
from utils.query_engine import RETRIEVAL_MODES

RETRIEVER_CHOICES = ("bm25", "dense", "hybrid")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrievers", default="bm25", help="Comma-separated: bm25,dense,hybrid")
    parser.add_argument("--max-articles", type=int, default=None, help="Use only the first N of 48 articles")
    parser.add_argument("--max-questions", type=int, default=None, help="Sample N questions (seeded)")
    parser.add_argument("--ks", default="1,3,5,10", help="Cutoffs for recall@k; MRR uses the largest")
    # Defaults follow the app's own config, .env included, so runs measure what the app ships.
    load_dotenv()
    app = AppConfig.from_env()
    parser.add_argument("--chunk-size", type=int, default=app.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=app.chunk_overlap)
    parser.add_argument("--embedding-model", default=app.embedding_model)
    parser.add_argument("--faithfulness", type=int, default=0, metavar="N",
                        help="Generate and judge N answers through the app's QueryEngine (requires dense)")
    parser.add_argument("--llm-model", default=app.llm_model, help="Answer model")
    parser.add_argument("--temperature", type=float, default=app.llm_temperature, help="Answer temperature")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--retrieval-mode", choices=RETRIEVAL_MODES, default=app.retrieval_mode,
                        help="Retriever the app's QueryEngine uses for the faithfulness run")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=None, help="Write results JSON here")
    args = parser.parse_args(argv)

    args.retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    unknown = set(args.retrievers) - set(RETRIEVER_CHOICES)
    if unknown:
        parser.error(f"Unknown retrievers: {sorted(unknown)}")
    if not args.retrievers:
        parser.error("--retrievers needs at least one of " + ",".join(RETRIEVER_CHOICES))
    args.ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    if not args.ks or args.ks[0] < 1:
        parser.error("--ks needs positive cutoffs")
    for flag in ("max_articles", "max_questions"):
        value = getattr(args, flag)
        if value is not None and value < 1:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    if args.faithfulness < 0:
        parser.error("--faithfulness must be zero or positive")
    if args.faithfulness and "dense" not in args.retrievers:
        parser.error("--faithfulness runs the app's QueryEngine on the dense index; add dense to --retrievers")
    return args


def evaluate_retrieval(retriever, questions, query_ids, relevant, ks) -> Dict:
    max_k = max(ks)
    started = time.perf_counter()
    rankings = retriever.retrieve_batch(questions, max_k)
    elapsed = time.perf_counter() - started

    metrics = {
        f"recall@{k}": mean(recall_at_k(r, relevant[qid], k) for r, qid in zip(rankings, query_ids))
        for k in ks
    }
    metrics[f"mrr@{max_k}"] = mean(
        reciprocal_rank(r, relevant[qid], max_k) for r, qid in zip(rankings, query_ids)
    )
    metrics["ms_per_query"] = 1000 * elapsed / len(questions)
    return metrics


def evaluate_faithfulness(
    dense: DenseRetriever, queries, relevant, args, answer_llm=None, judge_llm=None
) -> Dict:
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI

    from evaluation.faithfulness import judge_faithfulness
    from utils.query_engine import QueryEngine

    # QueryEngine reads the global Settings, as it does inside the app.
    Settings.llm = answer_llm or OpenAI(model=args.llm_model, temperature=args.temperature)
    Settings.embed_model = dense.embed_model
    judge = judge_llm or OpenAI(model=args.judge_model, temperature=0.0)
    # Same engine and retriever the app builds; hybrid fuses BM25 over the stored chunks
    engine = QueryEngine(dense.index, retrieval_mode=args.retrieval_mode)

    sample = random.Random(args.seed).sample(queries, min(args.faithfulness, len(queries)))
    examples = []
    for i, query in enumerate(sample, start=1):
        response = engine.query(query.question)
        contexts = [n.node.get_content() for n in response.source_nodes]
        source_ids = [n.node.node_id for n in response.source_nodes]
        result = judge_faithfulness(judge, query.question, response.response, contexts)
        examples.append({
            "query_id": query.query_id,
            "question": query.question,
            "gold_answers": list(query.answers),
            "answer": response.response,
            "source_chunk_ids": source_ids,
            "context_has_answer": any(cid in relevant[query.query_id] for cid in source_ids),
            "faithfulness": result.score,
            "claims": result.claims,
            "judge_invalid": result.invalid,
            "answer_recall": answer_recall(response.response, query.answers),
            "token_f1": token_f1(response.response, query.answers),
        })
        print(f"  faithfulness {i}/{len(sample)}", end="\r", file=sys.stderr)

    scored = [e["faithfulness"] for e in examples if e["faithfulness"] is not None]
    return {
        "summary": {
            "n": len(examples),
            # None, not NaN, when nothing was scorable, so the JSON stays strict
            "faithfulness": mean(scored) if scored else None,
            "fully_faithful_rate": mean(1.0 if s == 1.0 else 0.0 for s in scored) if scored else None,
            "no_claim_answers": sum(1 for e in examples if e["faithfulness"] is None and not e["judge_invalid"]),
            "judge_invalid": sum(1 for e in examples if e["judge_invalid"]),
            "answer_recall": mean(e["answer_recall"] for e in examples),
            "token_f1": mean(e["token_f1"] for e in examples),
            "context_has_answer": mean(1.0 if e["context_has_answer"] else 0.0 for e in examples),
            "retrieval_mode": args.retrieval_mode,
            "llm_model": args.llm_model,
            "temperature": args.temperature,
            "judge_model": args.judge_model,
        },
        "examples": examples,
    }


def format_table(results: Dict[str, Dict]) -> str:
    columns = [c for c in next(iter(results.values())) if c != "ms_per_query"]
    lines = [
        "| Retriever | " + " | ".join(columns) + " | ms/query |",
        "|---|" + "---|" * (len(columns) + 1),
    ]
    for name, metrics in results.items():
        cells = " | ".join(f"{metrics[c]:.3f}" for c in columns)
        lines.append(f"| {name} | {cells} | {metrics['ms_per_query']:.1f} |")
    return "\n".join(lines)


def main(argv: List[str] = None) -> Dict:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    dataset = load_squad(max_articles=args.max_articles, max_questions=args.max_questions, seed=args.seed)
    chunks = chunk_documents(dataset.documents, args.chunk_size, args.chunk_overlap)
    relevant = label_relevant_chunks(dataset.queries, chunks)
    queries = [q for q in dataset.queries if relevant[q.query_id]]
    questions = [q.question for q in queries]
    query_ids = [q.query_id for q in queries]
    print(
        f"Corpus: {len(dataset.documents)} documents, {len(chunks)} chunks. "
        f"Questions: {len(queries)} evaluated, "
        f"{len(dataset.queries) - len(queries)} skipped (answer split across chunks).",
        file=sys.stderr,
    )
    if not queries:
        sys.exit("No questions left to evaluate; try more articles or questions.")

    built = {}
    if "bm25" in args.retrievers or "hybrid" in args.retrievers:
        built["bm25"] = BM25Retriever(chunks)
    if "dense" in args.retrievers or "hybrid" in args.retrievers:
        built["dense"] = DenseRetriever(chunks, embedding_model=args.embedding_model)
    if "hybrid" in args.retrievers:
        built["hybrid"] = HybridRetriever([built["bm25"], built["dense"]])

    retrieval = {}
    for name in args.retrievers:
        print(f"Evaluating {name}...", file=sys.stderr)
        retrieval[name] = evaluate_retrieval(built[name], questions, query_ids, relevant, args.ks)

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "dataset": "squad-v1.1-dev",
            "documents": len(dataset.documents),
            "chunks": len(chunks),
            "questions_evaluated": len(queries),
            "questions_skipped": len(dataset.queries) - len(queries),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "embedding_model": args.embedding_model if "dense" in built else None,
            "seed": args.seed,
        },
        "retrieval": retrieval,
    }
    print("\n" + format_table(retrieval))

    if args.faithfulness:
        faithfulness = evaluate_faithfulness(built["dense"], queries, relevant, args)
        output["faithfulness"] = faithfulness
        print(f"\nAnswer quality ({args.retrieval_mode}, top-3, app QueryEngine):")
        print(json.dumps(faithfulness["summary"], indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)
    return output


if __name__ == "__main__":
    main()
