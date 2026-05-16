# Final Evaluation Protocol

## Goal

Evaluate the full Infineon KGQA system on three separate behaviors:

1. **KGQA answer quality**
2. **Request routing**
3. **Clarification triggering**

These should be reported separately. A single combined score hides different failure modes.

## Required datasets

### 1. KGQA benchmark

- Target size: **300-500 questions**
- Must contain gold SPARQL queries
- Must be frozen before final tuning
- Must not reuse questions inspected during tuning
- Should cover all major graph families:
  - regional demand
  - current demand / BL baselines
  - future demand
  - vehicle sales
  - autonomous driving
  - order cancellation
  - shortage
  - inventory
  - graph/catalog lookups
- Should cover multiple answer shapes:
  - raw values
  - grouped summaries
  - ranking / top-k
  - counts
  - comparisons

Primary metrics:

- Top-1 accuracy
- Any-correct candidate recall
- Selection gap: `Any Correct - Top-1`
- Generation failures
- Gold invalid / timeout counts
- Per-family breakdown

### 2. Routing benchmark

- Target size: **50-100 questions**
- Current dataset already has 60 examples
- Routes:
  - `definition`
  - `general_definition`
  - `clarification_needed`
  - `out_of_domain`
  - `kg_query`

Primary metrics:

- Routing accuracy
- Per-route confusion counts

### 3. Clarification benchmark

- Target size: **50-100 questions**
- Current dataset already has 60 examples
- Must include both:
  - questions that should ask for clarification
  - explicit questions that should not

Primary metrics:

- Clarification accuracy
- False-positive rate
- False-negative rate
- Per-topic breakdown

## Dataset hygiene

- `infineon_test_final.json` has already been inspected during tuning and should no longer be treated as the final unseen test.
- Existing inspected failures can be promoted into development analysis.
- The next final KGQA benchmark should be stored separately and kept frozen after creation.
- Any code changes after seeing failures on the new final benchmark must be reported as post-test tuning, not as final-test performance.

## Stop criteria

The system is ready for final reporting when:

- KGQA benchmark is frozen and large enough
- Top-1 accuracy is stable across repeated runs
- Candidate recall is reported alongside Top-1
- Clarification accuracy is at least acceptable and false positives are low enough for usability
- Routing accuracy is high enough that unsupported requests do not silently become graph queries
- Remaining failures are categorized, not anecdotal

## Recommended next target

Before more model tuning:

1. Build the new frozen KGQA benchmark.
2. Run the unified system report.
3. Perform error analysis on the new benchmark only after the first frozen run is saved.
