from evaluation.build_kgqa_seed_bank import build_seed_bank, summarize_seed_bank


def test_build_seed_bank_deduplicates_templates_and_maps_families(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        """[
          {"id":"A","question":"Show future demand by quarter.","query":"SELECT ?q (AVG(?v) AS ?avg) WHERE { ?x a survey:FutureDemandAnalysis ; survey:percentageChange ?v . } GROUP BY ?q","ambiguity_label":"high"},
          {"id":"B","question":"Give future demand by quarter.","query":"SELECT ?quarter (AVG(?pct) AS ?avg) WHERE { ?entry a survey:FutureDemandAnalysis ; survey:percentageChange ?pct . } GROUP BY ?quarter","ambiguity_label":"high"},
          {"id":"C","question":"Count shortages.","query":"SELECT (COUNT(?x) AS ?count) WHERE { ?x survey:hasShortage true . }","ambiguity_label":"low"}
        ]""",
        encoding="utf-8",
    )

    rows = build_seed_bank([str(dataset)])

    assert len(rows) == 2
    assert {row["family"] for row in rows} == {"future_demand", "shortage"}
    assert {row["answer_shape"] for row in rows} == {"average", "count"}


def test_summarize_seed_bank_builds_family_shape_matrix():
    rows = [
        {"family": "future_demand", "answer_shape": "average"},
        {"family": "future_demand", "answer_shape": "sum"},
        {"family": "shortage", "answer_shape": "count"},
    ]

    summary = summarize_seed_bank(rows)

    assert summary["families"] == {"future_demand": 2, "shortage": 1}
    assert summary["family_shape_matrix"]["future_demand"]["average"] == 1
    assert summary["family_shape_matrix"]["future_demand"]["sum"] == 1


def test_ranking_query_takes_priority_over_inner_count_shape(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        """[
          {"id":"A","question":"Which group has the most shortages?","query":"SELECT ?g (COUNT(?x) AS ?count) WHERE { ?x ?p ?g } GROUP BY ?g ORDER BY DESC(?count) LIMIT 1"}
        ]""",
        encoding="utf-8",
    )

    rows = build_seed_bank([str(dataset)])

    assert rows[0]["answer_shape"] == "ranking_top"


def test_future_demand_takes_family_priority_over_option_baseline_terms(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        """[
          {"id":"A","question":"Compare future demand Option1 and Option2 by quarter.","query":"SELECT ?q WHERE { ?x a survey:FutureDemandAnalysis ; survey:baselineType ?b . }"}
        ]""",
        encoding="utf-8",
    )

    rows = build_seed_bank([str(dataset)])

    assert rows[0]["family"] == "future_demand"
