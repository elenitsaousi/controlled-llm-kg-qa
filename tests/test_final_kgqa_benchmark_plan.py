from evaluation.build_final_kgqa_benchmark_plan import build_plan


def test_build_plan_applies_family_quotas_and_balances_reuse():
    seeds = []
    families = [
        "regional_demand",
        "current_demand_baselines",
        "future_demand",
        "vehicle_sales",
        "autonomous_driving",
        "order_cancellation",
        "shortage",
        "inventory",
        "catalog_lookup",
    ]
    for family in families:
        for idx in range(2):
            seeds.append(
                {
                    "template_id": f"{family}_{idx}",
                    "family": family,
                    "answer_shape": "sum" if idx == 0 else "count",
                    "ambiguity_label": "low" if idx == 0 else "high",
                    "source_id": f"{family}_{idx}",
                    "example_question": "q",
                    "query": "SELECT * WHERE {}",
                }
            )

    plan = build_plan(seeds)

    assert plan["summary"]["total_questions"] == 360
    assert plan["summary"]["families"]["future_demand"] == 50
    assert plan["summary"]["families"]["vehicle_sales"] == 45
    assert plan["summary"]["max_reuse_per_template"] == 25
    future_rows = [row for row in plan["rows"] if row["family"] == "future_demand"]
    assert future_rows[0]["template_id"] != future_rows[1]["template_id"]
    assert future_rows[2]["variant_index"] == 2


def test_build_plan_tracks_target_ambiguity_counts_independent_of_seed_labels():
    seeds = []
    for family in [
        "regional_demand",
        "current_demand_baselines",
        "future_demand",
        "vehicle_sales",
        "autonomous_driving",
        "order_cancellation",
        "shortage",
        "inventory",
        "catalog_lookup",
    ]:
        for idx in range(3):
            seeds.append(
                {
                    "template_id": f"{family}_{idx}",
                    "family": family,
                    "answer_shape": "sum",
                    "ambiguity_label": "low",
                    "source_id": f"{family}_{idx}",
                    "example_question": "q",
                    "query": "SELECT * WHERE {}",
                }
            )

    plan = build_plan(seeds)

    assert plan["summary"]["ambiguity"] == {"low": 108, "mid": 126, "high": 126}
