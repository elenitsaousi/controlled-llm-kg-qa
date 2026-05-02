# Infineon Data Splits

Use these canonical files for the current branch:

- `infineon_train.json`: 300 examples used to train the query-plan predictor and any ML ranker.
- `infineon_dev.json`: 100 examples used for prompt, template, ranking, and execution-selection tuning.
- `infineon_test_final.json`: 50 examples reserved for final evaluation. Do not tune code against this file; after inspecting its failures, promote it to dev and create a new final test.

The graph and schema inputs remain:

- `graph.ttl`
- `schema.json`
- `ontology.ttl`

Older generated datasets and previous evaluation sets live in `archive/` so the active data directory stays small and the split roles are explicit.
