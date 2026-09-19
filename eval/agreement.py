"""
Cohen's kappa, for measuring how well the model judge agrees with human
labels.

Raw agreement is the obvious number and the misleading one. If 85% of
answers are grounded, a judge that blindly answers "grounded" every time
agrees with a human 85% of the time while containing no information at
all. Kappa subtracts the agreement you would expect from two raters
guessing with the same class frequencies, so that degenerate judge
scores near zero, which is what it deserves.

    kappa = (observed - expected) / (1 - expected)

Rough convention for reading the result: below 0.2 is negligible, 0.2 to
0.4 fair, 0.4 to 0.6 moderate, 0.6 to 0.8 substantial, above 0.8 near
perfect. These bands are a convention rather than a law, and on a sample
of 50 the confidence interval is wide enough that the difference between
adjacent bands is not worth arguing about.

Unweighted kappa is used here because the three labels, grounded,
ungrounded and abstained, are nominal categories rather than points on a
scale. There is no sense in which abstained sits "between" the other
two, so there is no ordering for weights to encode.
"""

from collections import Counter


def cohen_kappa(labels_a, labels_b):
    """
    Agreement between two raters over the same items, corrected for
    chance. Returns a float in [-1, 1].

    Both sequences must be the same length and aligned item by item.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Rater label sequences must be the same length")
    if not labels_a:
        raise ValueError("Cannot compute agreement over zero items")

    total = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        (counts_a[label] / total) * (counts_b[label] / total)
        for label in set(counts_a) | set(counts_b)
    )

    if expected == 1.0:
        # Both raters used a single identical label for every item. They
        # agree completely, but chance agreement is also total, so kappa
        # is 0/0. Reporting 1.0 here would claim a perfect judge from
        # data containing no evidence either way.
        return 0.0

    return (observed - expected) / (1 - expected)


def confusion_matrix(labels_a, labels_b, label_names):
    """
    Counts of every (rater A, rater B) label pair.

    Worth printing next to kappa: a single number says how much the
    judge disagrees, and the matrix says which direction it fails in.
    A judge that never says "ungrounded" and one that says it randomly
    can produce the same kappa while being wrong in opposite, and very
    differently fixable, ways.
    """
    return {
        a: {b: sum(1 for x, y in zip(labels_a, labels_b) if x == a and y == b) for b in label_names}
        for a in label_names
    }
