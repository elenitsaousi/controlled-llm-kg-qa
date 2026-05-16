from evaluation.report_kgqa_seed_coverage import build_report


def test_build_report_marks_low_count_and_missing_shapes(tmp_path):
    seed_bank = tmp_path / "seed_bank.json"
    seed_bank.write_text(
        """[
          {"family":"inventory","answer_shape":"raw_or_lookup"},
          {"family":"inventory","answer_shape":"count"},
          {"family":"shortage","answer_shape":"raw_or_lookup"},
          {"family":"shortage","answer_shape":"count"},
          {"family":"shortage","answer_shape":"sum"},
          {"family":"shortage","answer_shape":"average"},
          {"family":"shortage","answer_shape":"ranking_top"},
          {"family":"shortage","answer_shape":"sum"},
          {"family":"shortage","answer_shape":"sum"},
          {"family":"shortage","answer_shape":"sum"},
          {"family":"shortage","answer_shape":"sum"},
          {"family":"shortage","answer_shape":"sum"}
        ]""",
        encoding="utf-8",
    )

    report = build_report(str(seed_bank))
    gaps = report["coverage_gaps"]

    assert any(gap["family"] == "inventory" and gap["gap_type"] == "low_template_count" for gap in gaps)
    assert any(gap["family"] == "inventory" and gap["gap_type"] == "low_shape_diversity" for gap in gaps)
    assert not any(gap["family"] == "shortage" and gap["gap_type"] == "low_template_count" for gap in gaps)
    inventory = next(row for row in report["family_rows"] if row["family"] == "inventory")
    assert inventory["final_question_quota"] == 35
    assert inventory["target_min_templates"] == 9
