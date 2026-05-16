from evaluation.build_kgqa_seed_bank import build_seed_bank


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
