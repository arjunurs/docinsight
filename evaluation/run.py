"""
Run the DocInsight evaluation.

Examples:
    # Offline lexical baseline, no API key needed
    python -m evaluation.run --retrievers bm25

    # Same on Natural Questions (real search queries, less word overlap)
    python -m evaluation.run --dataset nq --retrievers bm25

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
from evaluation.metrics import answer_recall, mean, percentile, reciprocal_rank, recall_at_k, token_f1
from evaluation.nq import load_nq
from evaluation.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever
from utils.config import AppConfig
from utils.query_engine import RETRIEVAL_MODES
from utils.reranking import build_reranker, reranker_key
from utils.retrieval import FUSION_MODES

RETRIEVER_CHOICES = ("bm25", "dense", "hybrid", "rerank")
PROMPTS = ("grounded", "default")
# name -> (loader taking max_documents, max_questions, seed; name recorded in results)
DATASETS = {
    "squad": (
        lambda max_documents, **kw: load_squad(max_articles=max_documents, **kw),
        "squad-v1.1-dev",
    ),
    "nq": (load_nq, "mrqa-nq-dev"),
}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="squad",
                        help="squad (SQuAD v1.1 dev) or nq (Natural Questions, MRQA dev)")
    parser.add_argument("--retrievers", default="bm25", help="Comma-separated: bm25,dense,hybrid")
    parser.add_argument("--max-articles", "--max-documents", dest="max_articles", type=int, default=None,
                        help="Use only the first N documents (48 SQuAD articles, 5,332 NQ passages)")
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
    parser.add_argument("--llm-model", default=app.llm_model,
                        help="Answer model, or a comma-separated list to compare on the same questions; "
                             "MODEL:EFFORT sets one model's reasoning effort (claude-... models use Anthropic)")
    parser.add_argument("--temperature", type=float, default=app.llm_temperature,
                        help="Answer temperature (older models only; reasoning models run at 1.0)")
    parser.add_argument("--reasoning-effort", default=app.llm_reasoning_effort,
                        help="Answer model reasoning effort: none, low, medium, ... (reasoning models only)")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--retrieval-mode", choices=RETRIEVAL_MODES, default=app.retrieval_mode,
                        help="Retriever the app's QueryEngine uses for the faithfulness run")
    parser.add_argument("--fusion", choices=FUSION_MODES, default=app.hybrid_fusion,
                        help="How hybrid fuses BM25 and dense: score (weighted, normalized scores) or rrf")
    parser.add_argument("--dense-weight", type=float, default=app.hybrid_dense_weight,
                        help="Weight on dense in score fusion, 0 to 1")
    parser.add_argument("--reranker", default=app.reranker, metavar="NAME",
                        help="Cross-encoder for the rerank retriever and the faithfulness run "
                             "(minilm; needs uv sync --extra rerank). Default: the app's RERANKER, off if unset")
    parser.add_argument("--rerank-candidates", type=int, default=app.rerank_candidates,
                        help="Candidates the reranker chooses from")
    parser.add_argument("--rerank-base", choices=("hybrid", "dense"), default="hybrid",
                        help="Retriever whose candidates the rerank retriever reorders")
    parser.add_argument("--prompt", choices=PROMPTS, default="grounded",
                        help="Answer prompt: the app's (grounded, says when the documents lack the answer) "
                             "or LlamaIndex's default, for comparison")
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
    if "rerank" in args.retrievers and not reranker_key(args.reranker):
        parser.error("--retrievers rerank needs --reranker (for example --reranker minilm)")
    # The app's QueryEngine never reranks fewer candidates than the 3 chunks it sends to the LLM
    if not 0.0 <= args.dense_weight <= 1.0:
        parser.error("--dense-weight must be between 0 and 1")
    if args.rerank_candidates < 3:
        parser.error("--rerank-candidates must be at least 3 (the chunks sent to the LLM)")
    args.answer_models = []
    for spec in (s.strip() for s in args.llm_model.split(",") if s.strip()):
        model, _, effort = spec.partition(":")
        args.answer_models.append((model, effort or args.reasoning_effort))
    if not args.answer_models:
        parser.error("--llm-model needs at least one model")
    if args.faithfulness and not {"dense", "hybrid", "rerank"} & set(args.retrievers):
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
    dense: DenseRetriever, queries, relevant, args, answer_llm=None, judge_llm=None, reranker=None,
    model: str = None, reasoning_effort: str = None,
) -> Dict:
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI

    from llama_index.core.prompts.default_prompt_selectors import (
        DEFAULT_REFINE_PROMPT_SEL,
        DEFAULT_TEXT_QA_PROMPT_SEL,
    )

    from evaluation.faithfulness import judge_faithfulness
    from utils.llm import UsageRecorder, build_llm, cost_usd
    from utils.prompts import is_no_answer
    from utils.query_engine import QueryEngine

    model = model or args.llm_model
    reasoning_effort = args.reasoning_effort if reasoning_effort is None else reasoning_effort
    # QueryEngine reads the global Settings, as it does inside the app.
    Settings.llm = answer_llm or build_llm(model, args.temperature, reasoning_effort)
    Settings.embed_model = dense.embed_model
    # The judge stays on gpt-4o-mini so faithfulness scores compare across answer models
    judge = judge_llm or OpenAI(model=args.judge_model, temperature=0.0)
    # Same engine and retriever the app builds; hybrid fuses BM25 over the stored chunks
    prompts = {}
    if args.prompt == "default":
        prompts = {"text_qa_template": DEFAULT_TEXT_QA_PROMPT_SEL, "refine_template": DEFAULT_REFINE_PROMPT_SEL}
    engine = QueryEngine(
        dense.index,
        retrieval_mode=args.retrieval_mode,
        reranker=reranker,
        rerank_candidates=args.rerank_candidates,
        fusion=args.fusion,
        dense_weight=args.dense_weight,
        **prompts,
    )

    sample = random.Random(args.seed).sample(queries, min(args.faithfulness, len(queries)))
    examples = []
    # Token usage of the answer model only; the judge's calls happen outside recording()
    recorder = UsageRecorder().attach()
    try:
        for i, query in enumerate(sample, start=1):
            with recorder.recording() as calls:
                started = time.perf_counter()
                response = engine.query(query.question)
                latency_ms = 1000 * (time.perf_counter() - started)
            input_tokens = sum(c.input_tokens for c in calls)
            output_tokens = sum(c.output_tokens for c in calls)
            call_costs = [cost_usd(model, c) for c in calls]
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
                "no_answer": is_no_answer(response.response),
                "faithfulness": result.score,
                "claims": result.claims,
                "judge_invalid": result.invalid,
                "answer_recall": answer_recall(response.response, query.answers),
                "token_f1": token_f1(response.response, query.answers),
                "latency_ms": latency_ms,
                "llm_calls": len(calls),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": sum(call_costs) if calls and None not in call_costs else None,
            })
            print(f"  faithfulness {i}/{len(sample)}", end="\r", file=sys.stderr)
    finally:
        # The handler is global; detach it even when a call fails so it does not outlive the run
        recorder.detach()

    scored = [e["faithfulness"] for e in examples if e["faithfulness"] is not None]
    latencies = sorted(e["latency_ms"] for e in examples)
    costs = [e["cost_usd"] for e in examples if e["cost_usd"] is not None]
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
            # Said "I don't know" although the gold chunk was retrieved (over-cautious)
            "no_answer_with_answer_in_context": sum(1 for e in examples if e["no_answer"] and e["context_has_answer"]),
            # Answered although the gold chunk was not retrieved (where hallucinations come from)
            "answered_without_answer_in_context": sum(
                1 for e in examples if not e["no_answer"] and not e["context_has_answer"]
            ),
            "no_answer_rate": mean(1.0 if e["no_answer"] else 0.0 for e in examples),
            # Wall clock of the app's query: retrieval plus the answer call(s)
            "ms_per_query": mean(latencies),
            "ms_per_query_p50": percentile(latencies, 50),
            "ms_per_query_p95": percentile(latencies, 95),
            "input_tokens_per_query": mean(e["input_tokens"] for e in examples),
            "output_tokens_per_query": mean(e["output_tokens"] for e in examples),
            # Answer model only, at list prices (utils/llm.py PRICES); None for unpriced models
            "cost_per_1k_queries_usd": 1000 * mean(costs) if costs and len(costs) == len(examples) else None,
            "retrieval_mode": args.retrieval_mode,
            "fusion": args.fusion,
            "dense_weight": args.dense_weight if args.fusion == "score" else None,
            "prompt": args.prompt,
            "reranker": args.reranker or None,
            "rerank_candidates": engine.rerank_candidates if engine.reranker is not None else None,
            "llm_model": model,
            "temperature": args.temperature,
            "reasoning_effort": reasoning_effort or None,
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


MODEL_COLUMNS = (
    ("answer_recall", "Answer recall", "{:.2f}"),
    ("no_answer_with_answer_in_context", "Wrong \"don't know\"", "{}"),
    ("answered_without_answer_in_context", "Answered w/o gold", "{}"),
    ("faithfulness", "Faithfulness", "{:.3f}"),
    ("ms_per_query_p50", "p50 ms", "{:.0f}"),
    ("ms_per_query_p95", "p95 ms", "{:.0f}"),
    ("output_tokens_per_query", "Out tokens", "{:.0f}"),
    ("cost_per_1k_queries_usd", "$ / 1k queries", "{:.3f}"),
)


def format_model_table(summaries: Dict[str, Dict]) -> str:
    lines = [
        "| Model | " + " | ".join(label for _, label, _ in MODEL_COLUMNS) + " |",
        "|---|" + "---|" * len(MODEL_COLUMNS),
    ]
    for name, summary in summaries.items():
        cells = [
            "n/a" if summary.get(key) is None else fmt.format(summary[key]) for key, _, fmt in MODEL_COLUMNS
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: List[str] = None) -> Dict:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Fail on an unknown --reranker or a missing extra before paying to embed the corpus
    reranker = build_reranker(args.reranker) if "rerank" in args.retrievers or args.faithfulness else None

    load, dataset_name = DATASETS[args.dataset]
    dataset = load(max_documents=args.max_articles, max_questions=args.max_questions, seed=args.seed)
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
    rerank_hybrid = "rerank" in args.retrievers and args.rerank_base == "hybrid"
    needs_hybrid = "hybrid" in args.retrievers or rerank_hybrid
    if "bm25" in args.retrievers or needs_hybrid:
        built["bm25"] = BM25Retriever(chunks)
    if "dense" in args.retrievers or needs_hybrid or "rerank" in args.retrievers:
        built["dense"] = DenseRetriever(chunks, embedding_model=args.embedding_model)
    if needs_hybrid:
        built["hybrid"] = HybridRetriever(
            [built["bm25"], built["dense"]], fusion=args.fusion, dense_weight=args.dense_weight
        )
    if "rerank" in args.retrievers:
        base = built[args.rerank_base]
        built["rerank"] = RerankRetriever(base, chunks, reranker, args.rerank_candidates)

    retrieval = {}
    for name in args.retrievers:
        print(f"Evaluating {name}...", file=sys.stderr)
        retrieval[name] = evaluate_retrieval(built[name], questions, query_ids, relevant, args.ks)

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "dataset": dataset_name,
            "documents": len(dataset.documents),
            "chunks": len(chunks),
            "questions_evaluated": len(queries),
            "questions_skipped": len(dataset.queries) - len(queries),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "embedding_model": args.embedding_model if "dense" in built else None,
            "seed": args.seed,
            "hybrid_fusion": (
                {"fusion": args.fusion, "dense_weight": args.dense_weight if args.fusion == "score" else None}
                if "hybrid" in built else None
            ),
            "rerank": (
                {"reranker": args.reranker, "base": args.rerank_base, "candidates": args.rerank_candidates}
                if "rerank" in built else None
            ),
        },
        "retrieval": retrieval,
    }
    print("\n" + format_table(retrieval))

    if args.faithfulness and len(args.answer_models) == 1:
        model, effort = args.answer_models[0]
        faithfulness = evaluate_faithfulness(
            built["dense"], queries, relevant, args, reranker=reranker, model=model, reasoning_effort=effort
        )
        output["faithfulness"] = faithfulness
        print(f"\nAnswer quality ({args.retrieval_mode}, {args.prompt} prompt, top-3, app QueryEngine):")
        print(json.dumps(faithfulness["summary"], indent=2))
    elif args.faithfulness:
        # Same sampled questions and retrieved chunks for every model; only the answer model changes
        output["faithfulness_by_model"] = {}
        for model, effort in args.answer_models:
            print(f"Answering with {model} (effort {effort or 'default'})...", file=sys.stderr)
            output["faithfulness_by_model"][f"{model}:{effort}" if effort else model] = evaluate_faithfulness(
                built["dense"], queries, relevant, args, reranker=reranker, model=model, reasoning_effort=effort
            )
        print(f"\nAnswer models ({args.retrieval_mode}, {args.prompt} prompt, top-3, judge {args.judge_model}):")
        print(format_model_table({k: v["summary"] for k, v in output["faithfulness_by_model"].items()}))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)
    return output


if __name__ == "__main__":
    main()
