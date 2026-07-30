# Repaired Full-System Accuracy Metrics

This report evaluates the final user-facing system on the repaired 1000-question benchmark.

## Headline Metrics

| Metric | Value |
|---|---:|
| Overall answer-level accuracy | 894/1000 (89.4%) |
| KG analytics accuracy | 87.0% |
| DR ontology definition accuracy | 100.0% |
| Advisory accuracy | 96.0% |
| Deterministic route accuracy | 503/514 (97.9%) |
| LLM fallback answer accuracy | 391/486 (80.5%) |
| Cold-start LLM calls | 486 (48.6% of questions) |
| Warm-cache new LLM calls | 35 (3.5% of questions) |
| Cold-start estimated cost | €97.20 vs €200.00 all-LLM |
| Warm-cache observed cost | €7.00 |

## Accuracy by Expected Route

| Route | Rows | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| `advisory` | 50 | 48 | 2 | 96.0% |
| `definition` | 150 | 150 | 0 | 100.0% |
| `kg_analytics` | 800 | 696 | 104 | 87.0% |

## Accuracy by System Mode

| Mode | Rows | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| `direct_graph_supported` | 514 | 503 | 11 | 97.9% |
| `llm_ranking` | 486 | 391 | 95 | 80.5% |

## Remaining Incorrect Answers by Difficulty

| Difficulty | Count |
|---|---:|
| `easy` | 7 |
| `hard` | 74 |
| `medium` | 25 |

## Remaining Incorrect Answers by Failure Family

| Failure family | Count |
|---|---:|
| `autonomous_driving_complex_grouping` | 24 |
| `current_demand_baseline_or_scope` | 22 |
| `future_demand_complex_dimension` | 16 |
| `vehicle_sales_metric_or_dimension` | 14 |
| `other_semantic_mismatch` | 10 |
| `regional_demand_scope_or_dimension` | 8 |
| `wrong_scope_or_grouping` | 5 |
| `advisory_not_synthesized` | 2 |
| `incomplete_comparison` | 2 |
| `inventory_scope_or_dimension` | 1 |
| `shortage_scope_or_shape` | 1 |
| `wrong_answer_shape` | 1 |
