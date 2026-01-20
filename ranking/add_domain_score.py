# ranking/add_domain_score.py

import json
import numpy as np
from domain_features import compute_domain_embedding


def cosine(a, b):
    """
    Compute cosine similarity between two vectors.
    A small epsilon is added to avoid division by zero.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ------------------------------------------------------------------
# Question-level domain embeddings
# These represent the expected domain intent of each question.
# For now, they are manually defined placeholders.
# Later, they can be replaced by learned question embeddings.
# ------------------------------------------------------------------
QUESTION_EMB = {
    "Q1": np.array([1.0, 1.0]),   # supply chain + manufacturing
    "Q2": np.array([1.0, 1.0]),   # cross-domain
    "Q3": np.array([0.5, 1.0])    # mostly manufacturing
}


# ------------------------------------------------------------------
# Candidate-level entity type mapping
# This provides domain-relevant information for each candidate query.
# At this stage, the mapping is manually defined.
# In later stages, this can be extracted automatically from Cypher queries.
# ------------------------------------------------------------------
QUERY_ENTITY_TYPES = {
    "Q1_C1": ["Supplier", "Product", "Yield"],
    "Q1_C2": ["Supplier", "Material", "Yield"],
    "Q1_C3": ["Supplier"],
    "Q1_C4": ["Product"],
    "Q1_C5": ["Supplier", "ProcessStep", "Yield"],

    "Q2_C1": ["Supplier", "ProcessStep"],
    "Q2_C2": ["Supplier", "Product"],
    "Q2_C3": ["Material"],
    "Q2_C4": ["Product"],
    "Q2_C5": ["Supplier", "ProcessStep"],

    "Q3_C1": ["Product"],
    "Q3_C2": ["Tool"],
    "Q3_C3": [],
    "Q3_C4": [],
    "Q3_C5": ["Product"]
}


# ------------------------------------------------------------------
# Load existing feature file (baseline features)
# ------------------------------------------------------------------
with open("ranking/features.json") as f:
    data = json.load(f)


# ------------------------------------------------------------------
# Compute domain score for each candidate query
# ------------------------------------------------------------------
for qid, items in data.items():
    for item in items:
        query_id = item["query_id"]

        # Get entity types associated with the candidate query
        entity_types = QUERY_ENTITY_TYPES.get(query_id, [])

        # Compute domain embedding for the candidate
        z_c = compute_domain_embedding(entity_types)

        # Get the expected domain embedding for the question
        z_q = QUESTION_EMB[qid]

        # Compute cosine similarity as domain consistency score
        domain_score = cosine(z_c, z_q)

        # Store the continuous domain score as a new feature
        item["features"]["domain_score"] = float(domain_score)


# ------------------------------------------------------------------
# Write extended feature file (does NOT overwrite the original one)
# ------------------------------------------------------------------
with open("ranking/features_domain.json", "w") as f:
    json.dump(data, f, indent=2)
