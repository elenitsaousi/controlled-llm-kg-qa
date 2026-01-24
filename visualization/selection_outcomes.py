# visualization/selection_outcomes.py

import numpy as np

def selected_rank(scores: np.ndarray, ground_truth_index: int) -> int:
    """
    Returns the rank (1-based) of the ground truth candidate
    based on descending score order.
    """
    sorted_indices = np.argsort(scores)[::-1]
    rank = np.where(sorted_indices == ground_truth_index)[0][0]
    return rank + 1
