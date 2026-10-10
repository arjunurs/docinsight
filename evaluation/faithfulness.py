"""
Answer faithfulness: is every claim in a generated answer supported by the
retrieved context?

Follows the RAGAS definition (Es et al., 2023, arXiv:2309.15217): split the
answer into atomic claims, check each against the context, and score
supported_claims / total_claims. Claim extraction and verification happen in a
single judge call to halve cost.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

JUDGE_PROMPT = """You are grading whether an answer is grounded in the provided context.

Step 1: Break the ANSWER into short, self-contained factual claims. Skip filler and \
statements that the information is unavailable.
Step 2: For each claim, decide if it is directly stated in or can be inferred from the \
CONTEXT alone. Do not use outside knowledge.

Respond with JSON only, in this shape:
{{"claims": [{{"claim": "<claim>", "supported": true}}]}}
Return {{"claims": []}} if the answer contains no factual claims.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


@dataclass(frozen=True)
class FaithfulnessResult:
    # None when the answer made no factual claims (e.g. "I don't know") or the
    # judge never returned a valid reply; `invalid` tells the two apart.
    score: Optional[float]
    claims: List[dict]
    invalid: bool = False


def parse_judge_output(text: str) -> List[dict]:
    """
    Pull the claims list out of the judge's reply, tolerating code fences.

    Raises ValueError on anything malformed (no "claims" list, a non-boolean
    "supported") rather than guessing, since a guess could inflate the score.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge returned no JSON object: {text[:200]!r}")
    payload = json.loads(match.group(0))
    claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(claims, list):
        raise ValueError(f"Judge reply has no claims list: {text[:200]!r}")
    parsed = []
    for c in claims:
        if not isinstance(c, dict) or not isinstance(c.get("supported"), bool):
            raise ValueError(f"Judge claim lacks a boolean 'supported': {c!r}")
        parsed.append({"claim": str(c.get("claim", "")), "supported": c["supported"]})
    return parsed


def judge_faithfulness(
    llm, question: str, answer: str, contexts: Sequence[str], retries: int = 1
) -> FaithfulnessResult:
    """
    Score one answer. `llm` is any LlamaIndex LLM (anything with .complete()).

    A malformed judge reply is retried; if it stays malformed the answer is
    marked invalid and left out of the score instead of being guessed.
    """
    context = "\n\n---\n\n".join(contexts)
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    for attempt in range(retries + 1):
        try:
            claims = parse_judge_output(llm.complete(prompt).text)
            break
        except ValueError:
            if attempt == retries:
                return FaithfulnessResult(score=None, claims=[], invalid=True)
    if not claims:
        return FaithfulnessResult(score=None, claims=[])
    supported = sum(1 for c in claims if c["supported"])
    return FaithfulnessResult(score=supported / len(claims), claims=claims)
