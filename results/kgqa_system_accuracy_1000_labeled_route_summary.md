# Full-System Route and Accuracy Breakdown

## Overall

| Total | Correct | Incorrect | Unclear | Accuracy |
|---:|---:|---:|---:|---:|
| 1000 | 859 | 141 | 0 | 85.9% |

## By System Mode

| system_mode | Total | Correct | Incorrect | Unclear | Accuracy |
|---|---:|---:|---:|---:|---:|
| `llm_ranking` | 552 | 439 | 113 | 0 | 79.5% |
| `direct_graph_supported` | 448 | 420 | 28 | 0 | 93.8% |

## By Selected Source

| selected_source | Total | Correct | Incorrect | Unclear | Accuracy |
|---|---:|---:|---:|---:|---:|
| `validated_retrieval` | 382 | 290 | 92 | 0 | 75.9% |
| `capability_inventory` | 262 | 234 | 28 | 0 | 89.3% |
| `infineon` | 167 | 146 | 21 | 0 | 87.4% |
| `digital_reference_ontology` | 150 | 150 | 0 | 0 | 100.0% |
| `advisory` | 36 | 36 | 0 | 0 | 100.0% |
| `template` | 3 | 3 | 0 | 0 | 100.0% |

## By Expected Route

| expected_route | Total | Correct | Incorrect | Unclear | Accuracy |
|---|---:|---:|---:|---:|---:|
| `unknown` | 800 | 661 | 139 | 0 | 82.6% |
| `definition` | 150 | 150 | 0 | 0 | 100.0% |
| `advisory` | 50 | 48 | 2 | 0 | 96.0% |

## By Topic

| topic | Total | Correct | Incorrect | Unclear | Accuracy |
|---|---:|---:|---:|---:|---:|
| `unknown` | 150 | 150 | 0 | 0 | 100.0% |
| `autonomous_driving` | 89 | 62 | 27 | 0 | 69.7% |
| `catalog_lookup` | 89 | 87 | 2 | 0 | 97.8% |
| `current_demand_baselines` | 89 | 59 | 30 | 0 | 66.3% |
| `future_demand` | 89 | 75 | 14 | 0 | 84.3% |
| `inventory` | 89 | 88 | 1 | 0 | 98.9% |
| `order_cancellation` | 89 | 89 | 0 | 0 | 100.0% |
| `regional_demand` | 89 | 79 | 10 | 0 | 88.8% |
| `shortage` | 89 | 57 | 32 | 0 | 64.0% |
| `vehicle_sales` | 88 | 65 | 23 | 0 | 73.9% |
| `advisory_current_demand` | 10 | 10 | 0 | 0 | 100.0% |
| `advisory_future_demand` | 10 | 8 | 2 | 0 | 80.0% |
| `advisory_shortage` | 10 | 10 | 0 | 0 | 100.0% |
| `advisory_technology_signal` | 10 | 10 | 0 | 0 | 100.0% |
| `advisory_vehicle_signal` | 10 | 10 | 0 | 0 | 100.0% |

## LLM Fallback Reasons

| Reason | Rows |
|---|---:|
| validated retrieval candidate selected | 382 |
| LLM-generated KG query selected | 165 |
| advisory/recommendation route | 2 |
| kept_selected_nonempty | 2 |
| selected_best_executable_candidate | 1 |
