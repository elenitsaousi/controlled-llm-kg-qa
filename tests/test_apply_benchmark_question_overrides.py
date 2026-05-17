from evaluation.apply_benchmark_question_overrides import apply_overrides


def test_apply_overrides_replaces_requested_questions():
    rows = [
        {"id": "A", "question": "old"},
        {"id": "B", "question": "keep"},
    ]

    updated = apply_overrides(rows, {"A": "new"})

    assert updated[0]["question"] == "new"
    assert updated[0]["qc_override"] is True
    assert updated[1] == rows[1]
