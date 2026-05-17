from evaluation.report_duplicate_questions import find_duplicates


def test_find_duplicates_reports_repeated_questions():
    rows = [
        {"id": "A", "template_id": "t1", "question": "What is demand?"},
        {"id": "B", "template_id": "t2", "question": " what   is demand? "},
        {"id": "C", "template_id": "t3", "question": "What is supply?"},
    ]

    report = find_duplicates(rows)

    assert report["summary"] == {
        "total": 3,
        "unique_questions": 2,
        "duplicate_groups": 1,
        "duplicate_rows": 1,
    }
    assert report["cases"][0]["ids"] == ["A", "B"]
