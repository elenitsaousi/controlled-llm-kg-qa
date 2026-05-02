# Infineon KGQA Error Analysis

## Summary

- Total questions: 50
- Top1 correct: 21 (0.420)
- Any correct candidate: 40 (0.800)
- Ranking failures with a correct candidate present: 19
- Generation failures without a correct candidate: 10

## Failure Categories

| Category | Count |
|---|---:|
| `correct` | 21 |
| `generation_failure_no_correct_candidate` | 10 |
| `ranking_failure_correct_candidate_not_top1` | 19 |

## Family Performance

| Family | Total | Top1 | Any | Main failures |
|---|---:|---:|---:|---|
| `regional_total_demand_all_origins` | 5 | 0.000 | 0.800 | generation_failure_no_correct_candidate=1, ranking_failure_correct_candidate_not_top1=4 |
| `autonomous_sae_year_vehicle` | 5 | 0.200 | 0.200 | correct=1, generation_failure_no_correct_candidate=4 |
| `order_cancellation_responses_by_tech` | 5 | 0.200 | 0.600 | correct=1, generation_failure_no_correct_candidate=2, ranking_failure_correct_candidate_not_top1=2 |
| `vehicle_sales_forecast_month` | 5 | 0.200 | 0.800 | correct=1, generation_failure_no_correct_candidate=1, ranking_failure_correct_candidate_not_top1=3 |
| `future_demand_tech_quarter` | 5 | 0.400 | 1.000 | correct=2, ranking_failure_correct_candidate_not_top1=3 |
| `oem_regional_demand` | 5 | 0.400 | 1.000 | correct=2, ranking_failure_correct_candidate_not_top1=3 |
| `autonomous_sae_top_vehicle` | 5 | 0.600 | 0.600 | correct=3, generation_failure_no_correct_candidate=2 |
| `future_demand_vehicle_quarter` | 5 | 0.600 | 1.000 | correct=3, ranking_failure_correct_candidate_not_top1=2 |
| `tier1_automotive_current_bl` | 5 | 0.800 | 1.000 | correct=4, ranking_failure_correct_candidate_not_top1=1 |
| `vehicle_sales_actual_month` | 5 | 0.800 | 1.000 | correct=4, ranking_failure_correct_candidate_not_top1=1 |

## Ranking Failures

| ID | Family | Correct rank | Heuristic missing concepts | Question |
|---|---|---:|---|---|
| `FINAL001` | `regional_total_demand_all_origins` | 1 | - | Break down total demand by region and by survey group: Tier1, OEM, and Semiconductor. |
| `FINAL002` | `regional_total_demand_all_origins` | 1 | - | Create a region table with demand totals for all three survey origins. |
| `FINAL003` | `regional_total_demand_all_origins` | 1 | - | Across regions, separate total demand into Tier1, OEM, and Semiconductor buckets. |
| `FINAL005` | `regional_total_demand_all_origins` | 1 | - | I want regional demand totals grouped by origin type across Tier1, OEM, and Semiconductor. |
| `FINAL006` | `oem_regional_demand` | 1 | - | Sort regions from highest to lowest OEM demand. |
| `FINAL008` | `oem_regional_demand` | 1 | - | Where is OEM demand largest by region? |
| `FINAL010` | `oem_regional_demand` | 1 | - | Aggregate OEM-origin demand and order the regions by total. |
| `FINAL013` | `tier1_automotive_current_bl` | 1 | - | In Automotive Tier1 current demand, summarize the two baseline changes. |
| `FINAL017` | `future_demand_tech_quarter` | 4 | - | Build a quarterly future-demand view by technology category. |
| `FINAL018` | `future_demand_tech_quarter` | 1 | - | How do future-demand percentages differ across technologies over time? |
| `FINAL020` | `future_demand_tech_quarter` | 2 | - | Give a technology-by-quarter matrix of future-demand percentage changes. |
| `FINAL021` | `future_demand_vehicle_quarter` | 1 | VehicleType | For each vehicle type, calculate quarterly average future-demand change. |
| `FINAL023` | `future_demand_vehicle_quarter` | 1 | - | How does future demand shift by vehicle category over the quarters? |
| `FINAL037` | `order_cancellation_responses_by_tech` | 1 | - | Group order-cancellation participant counts by technology category and response direction. |
| `FINAL040` | `order_cancellation_responses_by_tech` | 1 | - | Summarize cancellation response types for all technology categories. |
| `FINAL042` | `vehicle_sales_actual_month` | 3 | VehicleType | Return monthly totals for actual vehicle-sales observations. |
| `FINAL047` | `vehicle_sales_forecast_month` | 1 | FutureDemandAnalysis, VehicleType | Return monthly totals for forecast vehicle-sales observations. |
| `FINAL049` | `vehicle_sales_forecast_month` | 2 | VehicleType | Sum forecasted vehicle sales for each month in the graph. |
| `FINAL050` | `vehicle_sales_forecast_month` | 1 | FutureDemandAnalysis, VehicleType | Give forecast vehicle-sales totals grouped by month. |

## Generation Failures

| ID | Family | Heuristic missing concepts | Question |
|---|---|---|---|
| `FINAL004` | `regional_total_demand_all_origins` | - | For each Infineon region, report demand totals for each survey-origin class. |
| `FINAL027` | `autonomous_sae_top_vehicle` | - | Find the vehicle category with the strongest SAE Level 5 percentage. |
| `FINAL030` | `autonomous_sae_top_vehicle` | - | Give the highest-percentage SAE Level 5 vehicle type. |
| `FINAL031` | `autonomous_sae_year_vehicle` | - | Across years, average autonomous-driving percentages for each vehicle type. |
| `FINAL032` | `autonomous_sae_year_vehicle` | - | Show yearly autonomous development averages by vehicle category. |
| `FINAL033` | `autonomous_sae_year_vehicle` | - | How do autonomous-driving percentages change per vehicle type by year? |
| `FINAL035` | `autonomous_sae_year_vehicle` | - | Build a yearly vehicle-type summary for autonomous driving development. |
| `FINAL036` | `order_cancellation_responses_by_tech` | - | For each semiconductor technology category, total the order-cancellation responses by response type. |
| `FINAL039` | `order_cancellation_responses_by_tech` | - | Return order-cancellation counts split by technology and response label. |
| `FINAL048` | `vehicle_sales_forecast_month` | FutureDemandAnalysis | What are forecast vehicle units by time period? |

