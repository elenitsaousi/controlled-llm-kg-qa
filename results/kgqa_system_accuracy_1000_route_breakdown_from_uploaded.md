# KGQA Full-System Route Breakdown

Total questions: 1000
LLM calls: 549
Estimated cost: EUR 109.80
Baseline all-LLM cost: EUR 200.00
Cost reduction: 45.1%

| System mode | Rows | Share |
|---|---:|---:|
| llm_ranking | 552 | 55.2% |
| direct_graph_supported | 448 | 44.8% |

| Source | Rows |
|---|---:|
| validated_retrieval | 382 |
| capability_inventory | 262 |
| infineon | 167 |
| digital_reference_ontology | 150 |
| advisory | 36 |
| template | 3 |

| Expected route | System mode | Rows |
|---|---|---:|
| kg | llm_ranking | 550 |
| kg | direct_graph_supported | 250 |
| definition | direct_graph_supported | 150 |
| advisory | direct_graph_supported | 48 |
| advisory | llm_ranking | 2 |

Accuracy is not computed here because the correctness column in the audit CSV is still empty.
