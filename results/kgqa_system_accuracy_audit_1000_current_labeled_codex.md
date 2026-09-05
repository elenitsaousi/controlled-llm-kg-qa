# System Accuracy Audit

This report is for the engineering system view. It should be reported separately from selection accuracy on LLM-needed questions.

## Overall

| Labeled | Correct | Incorrect | Unclear | Accuracy | Denominator | Unlabeled |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 859 | 141 | 0 | 0.859 | 1000 | 0 |

## By Mode

| Mode | Labeled | Correct | Incorrect | Unclear | Accuracy | Denominator | Unlabeled |
|---|---:|---:|---:|---:|---:|---:|---:|
| `direct_graph_supported` | 448 | 420 | 28 | 0 | 0.938 | 448 | 0 |
| `llm_ranking` | 552 | 439 | 113 | 0 | 0.795 | 552 | 0 |

## Reporting Note

Use this as system-level accuracy only after manual labels are completed. Keep it separate from Top-1 selection accuracy on the LLM-needed subset.
