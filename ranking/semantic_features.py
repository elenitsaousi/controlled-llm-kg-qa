# ranking/semantic_features.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once at module level
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(question: str, query: str) -> float:
    """Semantic similarity between question and SPARQL query."""
    q_emb = model.encode([question])
    c_emb = model.encode([query])
    return float(cosine_similarity(q_emb, c_emb)[0][0])

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