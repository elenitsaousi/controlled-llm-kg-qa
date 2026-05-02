# Infineon KGQA Error Analysis

## Summary

- Total questions: 50
- Top1 correct: 28 (0.560)
- Any correct candidate: 40 (0.800)
- Ranking failures with a correct candidate present: 12
- Generation failures without a correct candidate: 10

## Failure Categories

| Category | Count |
|---|---:|
| `correct` | 28 |
| `generation_failure_no_correct_candidate` | 10 |
| `ranking_failure_correct_candidate_not_top1` | 12 |

## Family Performance

| Family | Total | Top1 | Any | Main failures |
|---|---:|---:|---:|---|
| `autonomous_sae_year_vehicle` | 5 | 0.200 | 0.200 | correct=1, generation_failure_no_correct_candidate=4 |
| `future_demand_tech_quarter` | 5 | 0.200 | 0.800 | correct=1, generation_failure_no_correct_candidate=1, ranking_failure_correct_candidate_not_top1=3 |
| `regional_total_demand_all_origins` | 5 | 0.200 | 0.800 | correct=1, generation_failure_no_correct_candidate=1, ranking_failure_correct_candidate_not_top1=3 |
| `tier1_automotive_current_bl` | 5 | 0.200 | 1.000 | correct=1, ranking_failure_correct_candidate_not_top1=4 |
| `order_cancellation_responses_by_tech` | 5 | 0.600 | 0.600 | correct=3, generation_failure_no_correct_candidate=2 |
| `vehicle_sales_forecast_month` | 5 | 0.600 | 0.800 | correct=3, generation_failure_no_correct_candidate=1, ranking_failure_correct_candidate_not_top1=1 |
| `autonomous_sae_top_vehicle` | 5 | 0.800 | 0.800 | correct=4, generation_failure_no_correct_candidate=1 |
| `oem_regional_demand` | 5 | 0.800 | 1.000 | correct=4, ranking_failure_correct_candidate_not_top1=1 |
| `future_demand_vehicle_quarter` | 5 | 1.000 | 1.000 | correct=5 |
| `vehicle_sales_actual_month` | 5 | 1.000 | 1.000 | correct=5 |

## Ranking Failures

| ID | Family | Correct rank | Heuristic missing concepts | Question |
|---|---|---:|---|---|
| `FINAL001` | `regional_total_demand_all_origins` | 2 | - | Break down total demand by region and by survey group: Tier1, OEM, and Semiconductor. |
| `FINAL003` | `regional_total_demand_all_origins` | 2 | - | Across regions, separate total demand into Tier1, OEM, and Semiconductor buckets. |
| `FINAL005` | `regional_total_demand_all_origins` | 3 | - | I want regional demand totals grouped by origin type across Tier1, OEM, and Semiconductor. |
| `FINAL008` | `oem_regional_demand` | 1 | - | Where is OEM demand largest by region? |
| `FINAL011` | `tier1_automotive_current_bl` | 1 | - | For Tier1 Automotive current demand, return the BL1 and BL2 percentage-change values. |
| `FINAL012` | `tier1_automotive_current_bl` | 1 | - | What does BL1 versus BL2 look like for Tier1 current demand in Automotive? |
| `FINAL014` | `tier1_automotive_current_bl` | 1 | - | Give baseline-level current-demand percentages for Tier1 Automotive, limited to BL1 and BL2. |
| `FINAL015` | `tier1_automotive_current_bl` | 1 | - | Compare the Tier1 Automotive current-demand baseline scenarios BL1 and BL2. |
| `FINAL016` | `future_demand_tech_quarter` | 1 | - | For every technology bucket and quarter, compute the mean future-demand change. |
| `FINAL019` | `future_demand_tech_quarter` | 1 | - | Return average future-demand change grouped by technology and quarter. |
| `FINAL020` | `future_demand_tech_quarter` | 1 | - | Give a technology-by-quarter matrix of future-demand percentage changes. |
| `FINAL047` | `vehicle_sales_forecast_month` | 1 | FutureDemandAnalysis, VehicleType | Return monthly totals for forecast vehicle-sales observations. |

## Generation Failures

| ID | Family | Heuristic missing concepts | Question |
|---|---|---|---|
| `FINAL004` | `regional_total_demand_all_origins` | DemandForRegion | For each Infineon region, report demand totals for each survey-origin class. |
| `FINAL017` | `future_demand_tech_quarter` | - | Build a quarterly future-demand view by technology category. |
| `FINAL030` | `autonomous_sae_top_vehicle` | - | Give the highest-percentage SAE Level 5 vehicle type. |
| `FINAL031` | `autonomous_sae_year_vehicle` | - | Across years, average autonomous-driving percentages for each vehicle type. |
| `FINAL032` | `autonomous_sae_year_vehicle` | - | Show yearly autonomous development averages by vehicle category. |
| `FINAL033` | `autonomous_sae_year_vehicle` | - | How do autonomous-driving percentages change per vehicle type by year? |
| `FINAL035` | `autonomous_sae_year_vehicle` | - | Build a yearly vehicle-type summary for autonomous driving development. |
| `FINAL036` | `order_cancellation_responses_by_tech` | - | For each semiconductor technology category, total the order-cancellation responses by response type. |
| `FINAL039` | `order_cancellation_responses_by_tech` | - | Return order-cancellation counts split by technology and response label. |
| `FINAL048` | `vehicle_sales_forecast_month` | FutureDemandAnalysis | What are forecast vehicle units by time period? |

