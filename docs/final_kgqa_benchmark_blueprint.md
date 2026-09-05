# Final KGQA Benchmark Blueprint

## Why a new benchmark is needed

The current active KGQA files contain 450 rows in total, but they are not yet a clean final benchmark:

- `infineon_test_final.json` has already been inspected during tuning.
- Several current `topic` labels describe dataset provenance rather than business content, for example `paraphrase_augmentation`, `heldout_eval`, and `gold_expansion`.
- The next final benchmark should therefore be built as a new frozen artifact with normalized business-family labels.

## Recommended target size

Use **360 questions** for the next frozen KGQA benchmark.

Reason:

- materially larger than the current 50-question final set
- large enough for per-family analysis
- still practical to execute repeatedly with the current live LLM pipeline

If runtime becomes acceptable later, extend the same design to 500 questions.

## Proposed family quotas

| Family | Target questions |
| --- | ---: |
| regional demand | 40 |
| current demand / baselines | 40 |
| future demand | 50 |
| vehicle sales | 45 |
| autonomous driving | 40 |
| order cancellation | 35 |
| shortages | 35 |
| inventory | 35 |
| catalog / graph lookups | 40 |
| **Total** | **360** |

## Proposed answer-shape quotas

| Answer shape | Target share |
| --- | ---: |
| grouped sum / total | 25-30% |
| grouped average | 15-20% |
| count | 15-20% |
| raw / lookup values | 15-20% |
| top / ranking | 10-15% |
| comparisons | 10-15% |

## Proposed ambiguity mix

| Ambiguity label | Target share |
| --- | ---: |
| low | 30% |
| mid | 35% |
| high | 35% |

The final ambiguity label should describe the question difficulty, not whether clarification is expected.

## Build rules

1. Every gold query must execute successfully against the graph and return non-empty results.
2. No final question should be tuned against before the frozen first run is saved.
3. Families must be semantic, not provenance-based.
4. Paraphrases may be used, but the benchmark should not be mostly paraphrases of a tiny number of gold query templates.
5. Questions should include both:
   - easy explicit wording
   - realistic user wording with domain language variation
6. Future-demand and BEV examples should not dominate the benchmark because they were heavily exercised during development.

## Immediate next construction step

Build a seed bank of validated gold query templates by family, then expand each family to quota using controlled paraphrases while preserving:

- family identity
- answer shape
- ambiguity label
- non-empty execution
