# Project Artifact Cleanup Guide

This guide defines which artifacts are canonical, which are historical
experiments, and which files are safe to remove locally. It is conservative:
do not delete benchmark inputs, final models, or thesis-cited reports unless a
new canonical artifact has replaced them.

## Cleanup Principle

Keep three categories separate:

1. **Runtime assets**: needed to run the Streamlit KGQA system.
2. **Scientific evaluation artifacts**: needed to support thesis claims.
3. **Historical experiments**: useful for traceability, but not part of the
   final narrative.

The most important rule is not to mix system-level accuracy with selection-only
accuracy. Keep both artifact sets, but report them separately.

## Canonical Runtime Assets

Keep these in the repository:

- `app.py`
- `pipeline/`, `llm/`, `kg/`, `ranking/`, `validation/`, `visualization/`, `ui/`
- `evaluation/` scripts
- `data/infineon/schema.json`
- `data/infineon/ontology.ttl`
- `data/infineon/true_demand_ontology_extracted.ttl`
- `data/infineon/true_demand_ontology_extracted.owl`
- `data/infineon/true_demand_webvowl.json`
- `evaluation/question_sets/true_demand_efficiency_500.json`
- `README.md`
- `docs/`

Keep `data/infineon/graph.ttl` only while this branch is private/internal. It is
the full True Demand KG used by local RDFLib and Fuseki. If the repository is
made public, remove the graph and provide a setup note instead.

## Canonical ML Selection Artifacts

Current final model:

- `ranking/models/final1000_wf_ranker_scope_origin.json`

Required training data:

- `ranking/final1000_wf_train_ranker_data.json`

Runtime fallback models currently referenced by `app.py`:

- `ranking/models/final1000_wf_ranker_shortage_grouped.json`
- `ranking/models/final1000_wf_ranker_shape_features.json`

These fallback models are useful because the app tries multiple model paths. If
you want a stricter final branch, update `app.py` first and then archive the
fallbacks.

## Archive Candidate Models

These are historical experiments. Archive them outside git or in a clearly
named local folder if they are not used in the thesis narrative:

- `ranking/models/final1000_wf_ranker.json`
- `ranking/models/final1000_wf_ranker_catalog_status.json`
- `ranking/models/final1000_wf_ranker_output_vars.json`
- `ranking/models/final1000_wf_ranker_projection_grouping.json`
- `ranking/models/final1000_wf_xgb_depth2.pkl`
- `ranking/models/final1000_wf_xgb_ltr_pairwise.pkl`
- `ranking/models/final360_*`
- `ranking/models/infineon_ranker.joblib`
- `ranking/models/logistic_ranker.joblib`

Do not delete these immediately if you still need to show the development path
from earlier experiments. For the final thesis, one table listing rejected
models is enough; the branch does not need every old model file.

## Canonical Evaluation Artifacts

Keep the final reports that support the thesis claims:

### Selection-Only Evaluation

Keep on the machine where they exist:

- `results/final1000_wf_test_eval_schema_no_ml.json`
- `results/final1000_wf_test_scope_origin_m010.json`
- `results/final1000_wf_test_entropy_regime_schema_vs_ml.json`
- `results/final1000_wf_test_entropy_regime_schema_vs_ml.md`
- `results/final1000_wf_test_entropy_regime_diagnostics.json`
- `results/final1000_wf_test_entropy_regime_diagnostics.md`
- `results/final1000_wf_test_scope_origin_confidence_routing_sorted_v2.json`
- `results/final1000_wf_test_scope_origin_confidence_routing_sorted_v2.md`

These are the files behind:

- baseline/schema selection vs ML selection
- entropy-regime analysis
- confidence routing
- Any-Correct vs forced Top-1 selection

### System-Level Evaluation

Keep:

- `evaluation/question_sets/true_demand_efficiency_500.json`
- `logs/kgqa_system_accuracy_500.jsonl` if you need traceability for the 500-run
- `results/kgqa_system_accuracy_audit_500.csv`
- `results/kgqa_system_accuracy_audit_500_v2_labeled.csv`
- `results/kgqa_system_accuracy_audit_500_v2_labeled.json`
- `results/kgqa_system_accuracy_audit_500_v2_labeled.md`
- `results/kgqa_system_accuracy_audit_500_v2_review_needed.csv`
- `results/kgqa_efficiency_500_after_direct_report.json`
- `results/kgqa_efficiency_500_after_direct_report.md`

These support the engineering view:

- deterministic direct routing accuracy
- fallback LLM/ranking accuracy
- overall system accuracy
- estimated LLM-call reduction and cost reduction

## Local Results That Are Historical

The following local result files are old 50-question or legacy experiments. They
can be archived if they are not cited:

- `results/infineon_eval_unseen_50_results.json`
- `results/infineon_holdout_eval_50.json`
- `results/infineon_test_final_results*.json`
- `results/infineon_test_final_*error_analysis*.json`
- `results/infineon_test_final_*error_analysis*.md`
- `results/infineon_query_plan_predictor_eval.json`
- `results/final_kgqa_benchmark_1000_plan.json`
- `results/final_kgqa_benchmark_1500_plan.json`
- `results/current_kgqa_coverage_audit.json`
- `results/current_kgqa_seed_bank.json`

Archive rather than delete if they document earlier thesis progress.

## Safe Local Deletes

These files are generated caches or build artifacts and can be removed locally:

- `.DS_Store`
- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`
- `data/infineon/__pycache__/`
- `evaluation/__pycache__/`
- `ranking/__pycache__/`
- `kg/__pycache__/`, `llm/__pycache__/`, `pipeline/__pycache__/`
- `ui/__pycache__/`, `validation/__pycache__/`, `visualization/__pycache__/`
- `true_demand_webvowl.json`
- `true_demand_webvowl_clean.json`
- `OWL2VOWL/` if you no longer need the local converter checkout
- `WebVOWL/` if you can rebuild or do not need the local frontend checkout

The canonical WebVOWL JSON is:

- `data/infineon/true_demand_webvowl.json`

The root-level `true_demand_webvowl*.json` files are duplicates/scratch exports.

## Suggested Archive Structure

Use a local folder outside git for old experiments:

```text
archive_local/
  2026-06-final360-experiments/
  2026-06-xgboost-experiments/
  2026-06-legacy-50q-results/
```

`archive_local/` should stay ignored.

## Windows Inventory Commands

List result files:

```powershell
Get-ChildItem results -File | Sort-Object Name | Select-Object Name, Length
```

List model files:

```powershell
Get-ChildItem ranking\models -File | Sort-Object Name | Select-Object Name, Length
```

Find likely historical model experiments:

```powershell
Get-ChildItem ranking\models -File |
  Where-Object { $_.Name -match 'final360|xgb|catalog_status|output_vars|projection_grouping' } |
  Sort-Object Name |
  Select-Object Name, Length
```

Find safe local caches:

```powershell
Get-ChildItem . -Recurse -Directory -Force |
  Where-Object { $_.Name -eq '__pycache__' -or $_.Name -eq '.pytest_cache' } |
  Select-Object FullName
```

## Cleanup Sequence

Recommended order:

1. Confirm the canonical final files exist on the Windows laptop.
2. Copy or archive historical experiments outside git.
3. Delete only local caches and duplicate root-level WebVOWL exports.
4. If you want a clean branch, remove tracked historical models/results in a
   separate commit titled `Archive historical experiment artifacts`.
5. Re-run the app smoke test:

```powershell
python -m streamlit run app.py
```

6. Re-run one direct-route question and one LLM/ranking fallback question.

## Reporting Rule

Use this distinction in the thesis:

- **Selection evaluation**: final1000 held-out benchmark, evaluates the hard
  LLM candidate-selection problem.
- **System evaluation**: 500 mixed practical questions, evaluates the complete
  engineering system with deterministic direct routing plus LLM fallback.

Do not report the 500-question system score as if it were the same metric as the
final1000 selection benchmark.
