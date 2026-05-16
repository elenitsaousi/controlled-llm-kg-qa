import pytest

from evaluation.run_system_scorecard import build_scorecard


def test_build_scorecard_combines_system_metrics(tmp_path):
    kgqa_path = tmp_path / "kgqa.json"
    routing_path = tmp_path / "routing.json"
    clarification_path = tmp_path / "clarification.json"

    kgqa_path.write_text(
        '{"summary":{"total":10,"top1_correct":7,"top1_correct_rate":0.7,'
        '"any_correct":9,"any_correct_rate":0.9,"gold_invalid":0,"gold_timeout":0,'
        '"llm_generation_failures":1}}',
        encoding="utf-8",
    )
    routing_path.write_text(
        '{"summary":{"total":5,"correct":4,"accuracy":0.8}}',
        encoding="utf-8",
    )
    clarification_path.write_text(
        '{"summary":{"total":4,"correct":2,"accuracy":0.5},"cases":['
        '{"expected_needs_clarification":true,"actual_needs_clarification":true},'
        '{"expected_needs_clarification":true,"actual_needs_clarification":false},'
        '{"expected_needs_clarification":false,"actual_needs_clarification":true},'
        '{"expected_needs_clarification":false,"actual_needs_clarification":false}]}',
        encoding="utf-8",
    )

    scorecard = build_scorecard(
        kgqa_results_path=str(kgqa_path),
        routing_report_path=str(routing_path),
        clarification_report_path=str(clarification_path),
    )

    assert scorecard["kgqa"]["selection_gap"] == pytest.approx(0.2)
    assert scorecard["routing"]["accuracy"] == 0.8
    assert scorecard["clarification"]["false_positives"] == 1
    assert scorecard["clarification"]["false_negatives"] == 1
