# Final Evaluation Runbook

This runbook reruns the current True Demand KGQA system after the latest direct-template, ontology, advisor, and execution-aware selection changes.

Keep the metrics separate:

- **System accuracy**: complete routed system over 500 mixed questions.
- **Selection accuracy**: LLM-needed held-out benchmark only.
- **Ontology QA accuracy**: deterministic Digital Reference ontology definition routing.
- **Efficiency**: paid LLM calls avoided by direct graph-supported routing.

## 0. Start Fuseki

Run this in a separate PowerShell window and keep it open:

```powershell
cd C:\Users\tsaousieleni\Downloads\apache-jena-fuseki-6.1.0\apache-jena-fuseki-6.1.0
.\fuseki-server.bat --file=C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\data\infineon\graph.ttl /infineon
```

Endpoint used below:

```text
http://localhost:3030/infineon/sparql
```

## 1. Validate Canonical Artifacts

```powershell
cd C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa
python evaluation\validate_canonical_artifacts.py --out-json results\canonical_artifact_validation_current.json --out-md results\canonical_artifact_validation_current.md
```

Expected: `error=0`.

## 2. KG System Accuracy and Efficiency

This evaluates the complete current system over 500 mixed KG questions:

- direct graph-supported templates,
- advisor/direct routes,
- LLM only for unresolved questions,
- execution-aware candidate selection.

This run may call the LLM for the non-direct subset.

```powershell
python evaluation\run_efficiency_question_set.py --questions evaluation\question_sets\true_demand_efficiency_500.json --out-log logs\kgqa_system_accuracy_500_current.jsonl --fuseki-query-url http://localhost:3030/infineon/sparql --call-llm --enable-llm-cache
```

Then create the audit sheet:

```powershell
python evaluation\build_system_accuracy_audit.py --log logs\kgqa_system_accuracy_500_current.jsonl --questions evaluation\question_sets\true_demand_efficiency_500.json --out-csv results\kgqa_system_accuracy_audit_500_current.csv
```

Fill `correctness` manually with one of:

```text
correct
incorrect
unclear
```

Then summarize it:

```powershell
python evaluation\build_system_accuracy_audit.py --labeled-csv results\kgqa_system_accuracy_audit_500_current.csv --out-json results\kgqa_system_accuracy_audit_500_current_labeled.json --out-md results\kgqa_system_accuracy_audit_500_current_labeled.md --unclear-as-incorrect
```

Efficiency/cost summary:

```powershell
python evaluation\analyze_system_efficiency.py --log logs\kgqa_system_accuracy_500_current.jsonl --cost-per-call 0.20 --out-json results\kgqa_efficiency_500_current_report.json --out-md results\kgqa_efficiency_500_current_report.md
```

Report this as the **engineering system view**, not as the selection benchmark.

## 3. LLM-Needed Selection Benchmark

This is the scientific query-selection evaluation. It should remain separate from the 500 mixed system questions.

### 3.1 Baseline: schema/semantic, no ML

If you want a fresh no-ML result:

```powershell
python evaluation\run_infineon_holdout_eval.py --dataset results\splits\final1000_within_family\test.json --k 8 --progress --query-timeout 10 --use-schema-ranking --resume --out results\final1000_wf_test_eval_schema_no_ml_current.json
```

Then analyze:

```powershell
python evaluation\analyze_infineon_results.py --results results\final1000_wf_test_eval_schema_no_ml_current.json --dataset results\splits\final1000_within_family\test.json --out-md results\final1000_wf_test_schema_no_ml_current_error_analysis.md
```

### 3.2 Train current ML ranker

Retrain after feature changes to avoid feature-vector mismatch:

```powershell
python ranking\train_infineon_np_tfidf_ranker.py --training-data ranking\final1000_wf_train_ranker_data.json --cv-out results\final1000_wf_train_ranker_current_cv.json --model-out ranking\models\final1000_wf_ranker_current.json
```

### 3.3 Apply current ML ranker

```powershell
python evaluation\apply_ml_ranker_to_results.py --results results\final1000_wf_test_eval_shape_features.json --model ranking\models\final1000_wf_ranker_current.json --schema data\infineon\schema.json --out results\final1000_wf_test_scope_origin_current_m010.json --guarded --min-margin 0.10 --min-score 0.45 --max-rank 4
```

Then analyze:

```powershell
python evaluation\analyze_infineon_results.py --results results\final1000_wf_test_scope_origin_current_m010.json --dataset results\splits\final1000_within_family\test.json --out-md results\final1000_wf_test_scope_origin_current_m010_error_analysis.md
```

Selection switch audit:

```powershell
python evaluation\analyze_selection_switches.py --before results\final1000_wf_test_eval_shape_features.json --after results\final1000_wf_test_scope_origin_current_m010.json --dataset results\splits\final1000_within_family\test.json --out-json results\final1000_wf_test_scope_origin_current_m010_switch_audit.json --out-md results\final1000_wf_test_scope_origin_current_m010_switch_audit.md
```

Failure diagnostics:

```powershell
python evaluation\analyze_selection_failures.py --results results\final1000_wf_test_scope_origin_current_m010.json --dataset results\splits\final1000_within_family\test.json --schema data\infineon\schema.json --out-json results\final1000_wf_test_scope_origin_current_selection_failures.json --out-md results\final1000_wf_test_scope_origin_current_selection_failures.md
```

## 4. Entropy / Ambiguity Analysis

```powershell
python evaluation\analyze_entropy_ambiguity.py --results results\final1000_wf_test_scope_origin_current_m010.json --dataset results\splits\final1000_within_family\test.json --score-key ml_score --sort-by-score --normalization softmax --temperature 0.10 --bucket-mode quantiles --out-json results\final1000_wf_test_scope_origin_current_entropy_softmax010.json --out-md results\final1000_wf_test_scope_origin_current_entropy_softmax010.md
```

Baseline vs ML by entropy regime:

```powershell
python evaluation\compare_entropy_regime_selection.py --baseline-results results\final1000_wf_test_eval_schema_no_ml_current.json --ml-results results\final1000_wf_test_scope_origin_current_m010.json --dataset results\splits\final1000_within_family\test.json --entropy-source ml --score-key ml_score --sort-by-score --normalization softmax --temperature 0.10 --bucket-mode quantiles --out-json results\final1000_wf_test_entropy_regime_schema_vs_ml_current.json --out-md results\final1000_wf_test_entropy_regime_schema_vs_ml_current.md
```

Diagnostic evidence:

```powershell
python evaluation\analyze_entropy_regime_diagnostics.py --baseline-results results\final1000_wf_test_eval_schema_no_ml_current.json --ml-results results\final1000_wf_test_scope_origin_current_m010.json --dataset results\splits\final1000_within_family\test.json --entropy-source ml --score-key ml_score --sort-by-score --normalization softmax --temperature 0.10 --bucket-mode quantiles --out-json results\final1000_wf_test_entropy_regime_diagnostics_current.json --out-md results\final1000_wf_test_entropy_regime_diagnostics_current.md
```

## 5. Digital Reference Ontology Benchmark

Build a deterministic ontology-definition benchmark from the local DR ontology:

```powershell
python evaluation\build_dr_ontology_benchmark.py --dr-ontology C:\Users\tsaousieleni\Downloads\dr\DigitalReference.ttl --limit 300 --out evaluation\question_sets\dr_ontology_benchmark_current.json
```

Run it:

```powershell
python evaluation\run_dr_ontology_benchmark.py --benchmark evaluation\question_sets\dr_ontology_benchmark_current.json --schema data\infineon\schema.json --dr-ontology C:\Users\tsaousieleni\Downloads\dr\DigitalReference.ttl --out-json results\dr_ontology_benchmark_current_report.json --out-md results\dr_ontology_benchmark_current_report.md
```

This should report `LLM calls: 0`; it evaluates deterministic ontology accessibility, not KG query selection.

## 6. Final Summary Table

After the above outputs exist, build one report table:

```powershell
python evaluation\build_final_evaluation_summary.py --system-accuracy results\kgqa_system_accuracy_audit_500_current_labeled.json --efficiency results\kgqa_efficiency_500_current_report.json --selection results\infineon_test_final_error_analysis.json --baseline-vs-ml results\final1000_wf_test_entropy_regime_schema_vs_ml_current.json --dr-ontology results\dr_ontology_benchmark_current_report.json --out-json results\final_evaluation_summary_current.json --out-md results\final_evaluation_summary_current.md
```

If you use a different ML analysis JSON, pass that path in `--selection`.

## Reporting Wording

Use this distinction in the thesis:

```text
Selection accuracy is measured only on the LLM-needed held-out benchmark and evaluates candidate ranking under ambiguity.
System accuracy is measured on the full 500-question routed system and includes deterministic graph-supported answers.
Ontology accuracy is measured separately on DR ontology definition questions and uses deterministic routing without LLM calls.
```

