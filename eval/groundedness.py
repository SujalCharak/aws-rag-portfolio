"""
Groundedness rubric, judge prompt, and verdict parsing.

Groundedness asks one narrow question: is every claim in the generated
answer supported by the passages that were retrieved for it. It is not
correctness, and it is not retrieval quality. An answer can be perfectly
grounded in passages that were the wrong ones to retrieve, and an answer
can be true in the world while inventing a claim the context never made.
Keeping those three apart is most of the value of measuring this at all,
which is why the rubric says so explicitly to both the model and the
human.

The rubric is defined once, here, and handed to the model judge and
printed on the human labelling sheet unchanged. If the two sides are
given different instructions then the agreement score measures the gap
between the instructions rather than the quality of the judge, and it
would still produce a number that looks meaningful.
"""

import re

VALID_LABELS = ("grounded", "ungrounded", "abstained")

RUBRIC = """Label the answer as exactly one of:

grounded    Every factual claim in the answer is supported by the numbered
            context passages.

ungrounded  The answer makes at least one factual claim the context does not
            support. This includes claims that are true in general but do not
            appear in these passages.

abstained   The answer declines to answer, saying it does not know or that the
            context does not contain the answer, without asserting facts of
            its own.

Judge only whether the context supports the answer. Do not judge whether the
answer is correct in the world. Do not judge whether the retrieved passages
were the right ones to retrieve."""

JUDGE_SYSTEM_PROMPT = f"""You are evaluating whether an answer is grounded in the context it was given.

{RUBRIC}

Reply with exactly two lines:
VERDICT: <grounded|ungrounded|abstained>
REASON: <one sentence>"""


def build_judge_prompt(question, context, answer):
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer to evaluate: {answer}"
    )


def parse_verdict(response_text):
    """
    Pull a label out of the judge's reply.

    Returns (label, reason). Label is None when the reply cannot be
    read, which is recorded rather than quietly coerced to a default.
    A judge that silently fails as "grounded" would inflate the score
    and look like a working system, which is the single worst failure
    mode available to an evaluation harness.
    """
    label = None
    reason = ""

    verdict_match = re.search(r"VERDICT:\s*(\w+)", response_text, re.IGNORECASE)
    if verdict_match:
        candidate = verdict_match.group(1).lower()
        if candidate in VALID_LABELS:
            label = candidate

    if label is None:
        # Fall back to a bare label anywhere in the reply, but only if
        # exactly one of the three appears. Two labels in one reply
        # means the judge hedged, and guessing which it meant would be
        # inventing data.
        found = {name for name in VALID_LABELS if re.search(rf"\b{name}\b", response_text, re.IGNORECASE)}
        if len(found) == 1:
            label = found.pop()

    reason_match = re.search(r"REASON:\s*(.+)", response_text, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    return label, reason
