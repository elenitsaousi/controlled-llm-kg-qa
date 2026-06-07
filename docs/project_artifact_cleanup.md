# Project Artifact Cleanup Guide

This guide separates files that should stay in the repository from generated experiment artifacts that can be archived locally. It is intentionally conservative: do not delete raw benchmark datasets, final selected results, final models, or reports used in the thesis.

## Keep In The Repository

These are source or documentation assets:

- `app.py`
- `pipeline/`
- `llm/`
- `kg/`
- `ranking/`
- `evaluation/`
- `validation/`
- `visualization/`
- `tests/`
- `data/infineon/schema.json`
- `data/infineon/graph.ttl` if the repo is expected to run offline without Fuseki setup
- `overview/`
- `docs/`

## Keep As Final Thesis Artifacts

These are important even if they are generated:

- final benchmark dataset:
  - `results/final_kgqa_benchmark_1000_repaired.json`
  - `results/splits/final1000_within_family/train.json`
  - `results/splits/final1000_within_family/dev.json`
  - `results/splits/final1000_within_family/test.json`
- final train/dev/test evaluations:
  - `results/final1000_wf_train_eval.json`
  - `results/final1000_wf_dev_eval.json`
  - `results/final1000_wf_test_eval_shape_features.json`
- final selected/reranked outputs:
  - `results/final1000_wf_dev_scope_origin_m010.json`
  - `results/final1000_wf_test_scope_origin_m010.json`
- final confidence reports:
  - `results/final1000_wf_test_scope_origin_confidence_routing_sorted_v2.json`
  - `results/final1000_wf_test_scope_origin_confidence_routing_sorted_v2.md`
- final selection and error analyses:
  - `results/final1000_wf_test_scope_origin_m010_error_analysis.md`
  - `results/final1000_wf_test_scope_origin_m010_switch_audit.md`
  - `results/final1000_wf_test_scope_origin_selection_failures.md`
  - `results/final1000_wf_test_high_confidence_mistakes.md`
- final ranker training data and model:
  - `ranking/final1000_wf_train_ranker_data.json`
  - `ranking/models/final1000_wf_ranker_scope_origin.json`

## Archive Rather Than Delete

Move old experiments to an archive folder if they are no longer used in the thesis narrative:

- `results/*final360*`
- `results/*xgb*`
- `results/*no_grouped*`
- `results/*no_contract*`
- `results/*grouped_features*`
- `results/*contract_features*`
- `results/*catalog_status*`
- `results/*projection_grouping*`
- `results/*shortage_grouped*`
- temporary sweep outputs where a better final output exists
- duplicate wording audit versions except the final repaired audit

Recommended archive location:

```text
archive/experiments_YYYYMMDD/
```

Keep archive folders outside the git-tracked thesis source unless you explicitly need to share them.

## Usually Safe To Delete Locally

These are runtime/cache files and can be regenerated:

- `__pycache__/`
- `.pytest_cache/`
- `*.pyc`
- `logs/kgqa_sessions.jsonl` after exporting any useful feedback
- `logs/kgqa_feedback.jsonl` after exporting any useful feedback

## Windows Inventory Commands

Run these from the project root on Windows to see what is taking space:

```powershell
Get-ChildItem results -File | Sort-Object Name | Select-Object Name, Length
Get-ChildItem results -File | Measure-Object Length -Sum
Get-ChildItem ranking\models -File | Sort-Object Name | Select-Object Name, Length
```

To list likely old experiment outputs without deleting:

```powershell
Get-ChildItem results -File |
  Where-Object { $_.Name -match 'final360|xgb|no_grouped|no_contract|grouped_features|contract_features|catalog_status|projection_grouping|shortage_grouped' } |
  Sort-Object Name |
  Select-Object Name, Length
```

To archive them after reviewing:

```powershell
New-Item -ItemType Directory -Force archive\experiments_old

Get-ChildItem results -File |
  Where-Object { $_.Name -match 'final360|xgb|no_grouped|no_contract|grouped_features|contract_features|catalog_status|projection_grouping|shortage_grouped' } |
  Move-Item -Destination archive\experiments_old
```

Do not run the archive command until the final thesis artifact list above exists locally.

## Naming Convention Going Forward

Use final names only for the result that should be cited:

- `final1000_*` for current benchmark artifacts
- `scope_origin` for the current final selector family
- `confidence_routing_sorted_v2` for selective-answering metrics

Use `scratch_`, `tmp_`, or date-stamped archive folders for experiments that are not final.
