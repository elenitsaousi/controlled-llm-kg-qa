from evaluation.audit_generated_benchmark_wording import audit_rows


def test_audit_rows_flags_measure_and_dimension_drift():
    rows = [
        {
            "id": "A",
            "family": "inventory",
            "answer_shape": "count",
            "source_question": "Show Tier1 inventory trend distribution by component.",
            "question": "How many Tier1 inventory records are there for each component over time?",
        },
        {
            "id": "B",
            "family": "future_demand",
            "answer_shape": "sum",
            "source_question": "Show total future demand percentage change by quarter.",
            "question": "What is the total projected demand by quarter?",
        },
    ]

    report = audit_rows(rows)

    assert report["summary"]["flagged"] == 2
    assert "added_time_dimension" in report["cases"][0]["warnings"]
    assert "lost_trend_dimension" in report["cases"][0]["warnings"]
    assert "lost_percentage_measure" in report["cases"][1]["warnings"]
    assert "lost_percentage_change_measure" in report["cases"][1]["warnings"]


def test_audit_rows_flags_added_and_lost_calendar_dimensions():
    rows = [
        {
            "id": "A",
            "family": "regional_demand",
            "answer_shape": "sum",
            "source_question": "Return total demand by region.",
            "question": "Return total demand by region and quarter?",
        },
        {
            "id": "B",
            "family": "vehicle_sales",
            "answer_shape": "sum",
            "source_question": "Return total vehicle sales by month.",
            "question": "Return total vehicle sales?",
        },
    ]

    report = audit_rows(rows)

    assert "added_quarter_dimension" in report["cases"][0]["warnings"]
    assert "lost_month_dimension" in report["cases"][1]["warnings"]


def test_audit_rows_skips_source_dependent_checks_when_source_missing():
    rows = [
        {
            "id": "A",
            "family": "regional_demand",
            "answer_shape": "sum",
            "source_question": "",
            "question": "How much demand did OEMs generate by quarter?",
        }
    ]

    report = audit_rows(rows)

    assert report["summary"]["flagged"] == 0


def test_audit_rows_accepts_ranking_and_year_synonyms():
    rows = [
        {
            "id": "A",
            "family": "vehicle_sales",
            "answer_shape": "ranking_top",
            "source_question": "Which vehicle type has the highest total yearly sales?",
            "question": "What vehicle type leads in annual sales?",
        }
    ]

    report = audit_rows(rows)

    assert report["summary"]["flagged"] == 0
