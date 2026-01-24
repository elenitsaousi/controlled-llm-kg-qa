# visualization/ambiguity_metrics.py

import numpy as np

def ambiguity_margin(scores: np.ndarray) -> float:
    """
    Ambiguity based on top-1 vs top-2 margin.

    A(q) = 1 - (s1 - s2) / s1

    High value => high ambiguity
    """
    sorted_scores = np.sort(scores)[::-1]
    s1, s2 = sorted_scores[0], sorted_scores[1]
    return 1.0 - (s1 - s2) / (s1 + 1e-9)


def ambiguity_entropy(scores: np.ndarray) -> float:
    """
    Entropy-based ambiguity.

    A(q) = - sum_j p_j log p_j
    where p_j = softmax(scores)
    """
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)
    return -np.sum(probs * np.log(probs + 1e-9))
