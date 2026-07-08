# Final 1000-Question Evaluation Protocol

This protocol separates two different evaluation claims:

1. **LLM-only query-selection benchmark**: measures candidate generation and selection on gold-query questions. Deterministic routing is disabled.
2. **Full-system benchmark**: measures the deployed architecture: deterministic graph-supported routing first, LLM only when direct routing cannot answer.

Do not merge these numbers without explanation. They answer different questions.

## A. LLM-Only Selection Benchmark

Goal:

- Run the LLM candidate generator once on the 1000-question gold benchmark.
- Measure whether the correct candidate appears in the candidate set (**Any Correct**).
- Compare selection policies offline on the same candidate set:
  - raw candidate order: no ML and no schema/semantic selector
  - schema/semantic selector
  - guarded ML reranker

Important:

- The LLM generation still receives schema context. Here "no schema" means no schema/semantic **selection** layer after generation, not a schema-free prompt.
- Use the same generated candidates for all selection policies. Otherwise the comparison is noisy and costs more.
- Keep the current ML model frozen for the first final evaluation:
  `ranking/models/final1000_wf_ranker_current.json`

Report:

| Mode | Questions | Top-1 | Any Correct | Ranking Failures | Generation Failures |
|---|---:|---:|---:|---:|---:|
| Raw LLM candidates | 1000 | from baseline analysis | from baseline analysis | from baseline analysis | from baseline analysis |
| Schema/semantic selector | 1000 | from replay analysis | same candidate set | from replay analysis | same candidate set |
| Guarded ML reranker | 1000 | from ML analysis | same candidate set | from ML analysis | same candidate set |

## B. Full-System Benchmark

Goal:

- Run the full system on a mixed 1000-question set.
- Deterministic routes answer first:
  - KG capability templates
  - DR ontology definitions
  - deterministic advisory routes
- Only unresolved questions call the LLM.

Default mixed composition:

| Source | Rows | Purpose |
|---|---:|---|
| True Demand KG/data questions | 800 | graph/data QA over survey-derived KG |
| DR ontology definition questions | 150 | ontology/model explanation without SPARQL |
| Advisory questions | 50 | deterministic graph-grounded business guidance |

Report:

| Mode | Rows | Accuracy | LLM Calls | Cost |
|---|---:|---:|---:|---:|
| Direct graph-supported | from audit | manual/gold audit | 0 | 0 |
| LLM/ranking fallback | from audit | manual/gold audit | from log | calls * 0.20 |
| Overall system | 1000 | full audit | from log | total calls * 0.20 |

## Correctness Requirement

For system-level accuracy, a row must be labeled using a gold expected query/result or a manual audit:

- `correct`: answer matches the requested metric, scope, grouping, and ranking intent.
- `incorrect`: wrong metric, wrong dimension, wrong scope, unsupported join, empty result presented as answer, or ranking when breakdown was requested.
- `unclear`: question itself is underspecified or answer is partially defensible.

For thesis reporting, show unclear handling explicitly, e.g. "unclear excluded" or "unclear counted as incorrect".

## Leakage / Retraining Note

Do not retrain the ML model on the final 1000 questions and then report performance on the same 1000 as held-out accuracy. If retraining is needed, create a train/test split first and report the test split only.

The first final evaluation should therefore use the frozen current model:

`ranking/models/final1000_wf_ranker_current.json`
