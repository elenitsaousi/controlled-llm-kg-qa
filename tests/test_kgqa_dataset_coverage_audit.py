from evaluation.audit_kgqa_dataset_coverage import audit


def test_audit_reports_topic_ambiguity_and_shape_counts(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        """[
          {"topic":"sales","ambiguity_label":"low","query":"SELECT (COUNT(?x) AS ?count) WHERE { ?x ?p ?o }"},
          {"topic":"sales","ambiguity_label":"mid","query":"SELECT ?x (SUM(?v) AS ?total) WHERE { ?x ?p ?v } GROUP BY ?x"},
          {"topic":"demand","ambiguity_label":"high","query":"SELECT ?x WHERE { ?x ?p ?o }"}
        ]""",
        encoding="utf-8",
    )

    report = audit([str(dataset)])

    assert report["total"] == 3
    assert report["topics"] == {"sales": 2, "demand": 1}
    assert report["ambiguity"] == {"low": 1, "mid": 1, "high": 1}
    assert report["answer_shapes"] == {"count": 1, "sum": 1, "raw_or_lookup": 1}
