# ranking/semantic_features.py
import os
import contextlib
import io
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _get_model():
    if os.getenv("ENABLE_SEMANTIC_EMBEDDING", "0") != "1":
        return None
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

def semantic_similarity(question: str, query: str) -> float:
    """Semantic similarity between question and SPARQL query."""
    model = _get_model()
    if model is None:
        return 0.0
    q_emb = np.asarray(model.encode([question])[0], dtype=float)
    c_emb = np.asarray(model.encode([query])[0], dtype=float)
    denom = float(np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(q_emb, c_emb) / denom)

def has_named_instance(query: str) -> int:
    """Check if query uses correct named instances [7]."""
    return int(any(inst in query for inst in [
        "Tier1CurrentDemand", "OEMCurrentDemand",
        "SemiCurrentDemand", "SemiFutureDemand_Option1",
        "SemiFutureDemand_Option2", "SemiFutureDemand_Option3",
    ]))

def uses_correct_survey(query: str) -> int:
    """Check if query uses correct survey origin [7]."""
    return int(any(s in query for s in [
        "OEM_Survey", "Tier1_Survey", "Semiconductor_Survey"
    ]))
