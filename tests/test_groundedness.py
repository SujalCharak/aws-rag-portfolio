"""
Tests for verdict parsing and rater agreement.

The kappa tests matter more than they look. Kappa is the number that
decides whether the groundedness judge is trustworthy enough to report
at all, and the failure it exists to catch, a judge that agrees often by
being constant, produces a high raw agreement score that looks like
success.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from agreement import cohen_kappa, confusion_matrix  # noqa: E402
from groundedness import parse_verdict  # noqa: E402


# --- verdict parsing ----------------------------------------------------

def test_parses_the_expected_two_line_format():
    label, reason = parse_verdict("VERDICT: grounded\nREASON: every claim appears in passage 2.")

    assert label == "grounded"
    assert reason == "every claim appears in passage 2."


def test_parsing_is_case_insensitive():
    label, _ = parse_verdict("verdict: UNGROUNDED\nreason: passage 1 says nothing about dosage.")

    assert label == "ungrounded"


def test_falls_back_to_a_bare_label_when_the_format_slips():
    label, _ = parse_verdict("This answer is abstained, it declines to answer.")

    assert label == "abstained"


def test_grounded_does_not_match_inside_ungrounded():
    """
    A substring match here would silently invert the verdict, turning
    every ungrounded answer into a grounded one and inflating the
    headline number.
    """
    label, _ = parse_verdict("VERDICT: ungrounded\nREASON: invented a statistic.")

    assert label == "ungrounded"


def test_a_hedged_reply_is_unparseable_rather_than_guessed():
    """
    Two labels in one reply means the judge did not commit. Picking one
    would be inventing data.
    """
    label, _ = parse_verdict("It is hard to say whether this is grounded or ungrounded.")

    assert label is None


def test_unreadable_reply_returns_none_not_a_default():
    """
    Defaulting an unparseable judge reply to 'grounded' would make a
    broken judge look like a working one, which is the worst failure
    available to an evaluation harness.
    """
    label, _ = parse_verdict("I'm sorry, I can't help with that.")

    assert label is None


def test_invalid_verdict_word_is_rejected():
    label, _ = parse_verdict("VERDICT: maybe\nREASON: unclear.")

    assert label is None


# --- agreement ----------------------------------------------------------

def test_kappa_matches_a_hand_computed_confusion_matrix():
    #            human:g  human:u
    #  judge:g      20       10      (30)
    #  judge:u       5       15      (20)
    #             (25)     (25)       50
    #
    #  observed = (20 + 15) / 50                     = 0.70
    #  expected = (30/50)(25/50) + (20/50)(25/50)    = 0.50
    #  kappa    = (0.70 - 0.50) / (1 - 0.50)         = 0.40
    judge = ["g"] * 30 + ["u"] * 20
    human = ["g"] * 20 + ["u"] * 10 + ["g"] * 5 + ["u"] * 15

    assert cohen_kappa(judge, human) == pytest.approx(0.40)


def test_a_constant_judge_scores_zero_despite_high_raw_agreement():
    """
    The entire reason for using kappa instead of raw agreement.

    This judge answers 'grounded' every time and is right 80% of the
    time, purely because 80% of the answers happen to be grounded. Raw
    agreement would report 0.80 and imply a useful judge. Kappa reports
    0.0, correctly, because the judge carries no information.
    """
    judge = ["grounded"] * 10
    human = ["grounded"] * 8 + ["ungrounded"] * 2

    raw_agreement = sum(1 for a, b in zip(judge, human) if a == b) / len(judge)

    assert raw_agreement == pytest.approx(0.80)
    assert cohen_kappa(judge, human) == pytest.approx(0.0)


def test_perfect_agreement_on_mixed_labels_is_one():
    labels = ["grounded", "ungrounded", "abstained", "grounded"]

    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_total_agreement_on_a_single_label_is_zero_not_one():
    """
    Both raters said 'grounded' for everything. They never disagreed,
    but chance agreement is also total, so there is no evidence the
    judge can tell the classes apart. Reporting 1.0 would claim a
    perfect judge from data that contains no information.
    """
    assert cohen_kappa(["grounded"] * 5, ["grounded"] * 5) == pytest.approx(0.0)


def test_systematic_disagreement_goes_negative():
    judge = ["grounded", "grounded", "ungrounded", "ungrounded"]
    human = ["ungrounded", "ungrounded", "grounded", "grounded"]

    assert cohen_kappa(judge, human) < 0


def test_mismatched_lengths_raise_rather_than_silently_truncating():
    with pytest.raises(ValueError):
        cohen_kappa(["grounded"], ["grounded", "ungrounded"])


def test_confusion_matrix_shows_the_direction_of_disagreement():
    judge = ["grounded", "grounded", "ungrounded"]
    human = ["grounded", "ungrounded", "ungrounded"]

    matrix = confusion_matrix(judge, human, ["grounded", "ungrounded"])

    assert matrix["grounded"]["grounded"] == 1
    assert matrix["grounded"]["ungrounded"] == 1
    assert matrix["ungrounded"]["ungrounded"] == 1
    assert matrix["ungrounded"]["grounded"] == 0


def test_kappa_matches_scikit_learn_when_available():
    """
    Cross check against the reference implementation, same idea as the
    trec_eval check on the retrieval metrics. Skips cleanly when
    scikit-learn is not installed, since it is a heavy dependency to
    require for one assertion.
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    judge = ["g"] * 30 + ["u"] * 20
    human = ["g"] * 20 + ["u"] * 10 + ["g"] * 5 + ["u"] * 15

    assert cohen_kappa(judge, human) == pytest.approx(
        sklearn_metrics.cohen_kappa_score(judge, human), abs=1e-9
    )
