**Formal Problem Statement + Constrained Decoding Objective**
Let x be a natural-language question and S the KG schema (labels, relations, property types).
Let Q_S be the set of schema-valid structured queries.

We seek:

\hat{q} = argmax_{q in Q_S} P_theta(q | x)

This enforces schema-compliance by construction. In practice, constrained decoding or
filtering ensures q is in Q_S.

**Ranking Model + Loss Functions**
The LLM produces k candidates Q = {q_1, ..., q_k}.
We learn a ranking score s_phi(x, q) and select:

\hat{q} = argmax_{q in Q} s_phi(x, q)

Supervised softmax loss:

L_softmax = -log( exp(s_phi(x, q*)) / sum_{q_i in Q} exp(s_phi(x, q_i)) )

Pairwise hinge loss (for weak supervision):

L_pair = sum_{(q+, q-)} max(0, 1 - s_phi(x, q+) + s_phi(x, q-))

Weak supervision signals:
- schema validity
- execution success
- semantic consistency checks
- answer non-emptiness (optional)

**Validation/Hallucination Penalties + Metrics**
Define constraint violations:

V(q) = V_syntax(q) + V_schema(q) + V_semantic(q)

Constrained selection:

\hat{q} = argmax_{q in Q} ( s_phi(x, q) - lambda * V(q) )

Hallucination categories (example):
- Schema hallucination: invalid labels/relations/properties
- Execution hallucination: query fails to execute
- Semantic hallucination: query executes but violates domain coherence

Metrics:
- Exact query match: (1/N) * sum I[\hat{q} = q*]
- Execution rate: (1/N) * sum I[q executes]
- Schema validity rate: (1/N) * sum I[V_schema(q) = 0]
- Hallucination rate: (1/N) * sum I[V(q) > 0]
