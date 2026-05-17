from evaluation.generate_final_kgqa_benchmark_llm import _normalize_measure_wording, _parse_question, generate_rows


class _FakeClient:
    def generate_text(self, _prompt):
        return '{"question":"How does demand vary by region?"}'


class _RetryClient:
    def __init__(self):
        self.responses = iter(
            [
                '{"question":"Which regions had the highest OEM demand in each quarter?"}',
                '{"question":"What is the total OEM demand by region and quarter?"}',
            ]
        )

    def generate_text(self, _prompt):
        return next(self.responses)


class _MalformedThenValidClient:
    def __init__(self):
        self.responses = iter(
            [
                '{"question": "For vehicles with Level 5 autonomy,',
                '{"question":"Which vehicle type has the highest percentage at Level 5 autonomy?"}',
            ]
        )

    def generate_text(self, _prompt):
        return next(self.responses)


def test_parse_question_accepts_json_object():
    assert _parse_question('{"question":"What is demand"}') == "What is demand?"


def test_normalize_measure_wording_replaces_typical_for_average_shape():
    assert (
        _normalize_measure_wording("What is the typical OEM demand?", "average")
        == "What is the average OEM demand?"
    )


def test_generate_rows_preserves_gold_query_and_target_label():
    plan = {
        "rows": [
            {
                "template_id": "t1",
                "family": "regional_demand",
                "answer_shape": "sum",
                "target_ambiguity_label": "high",
                "seed_ambiguity_label": "low",
                "source_id": "S1",
                "example_question": "Return total demand by region.",
                "query": "SELECT ...",
            }
        ]
    }

    rows = generate_rows(plan, client=_FakeClient())

    assert rows[0]["query"] == "SELECT ..."
    assert rows[0]["ambiguity_label"] == "high"
    assert rows[0]["question"] == "How does demand vary by region?"


def test_generate_rows_retries_when_wording_audit_detects_drift():
    plan = {
        "rows": [
            {
                "template_id": "t1",
                "family": "regional_demand",
                "answer_shape": "sum",
                "target_ambiguity_label": "mid",
                "seed_ambiguity_label": "mid",
                "source_id": "S1",
                "example_question": "Show OEM demand by region and quarter.",
                "query": "SELECT ...",
            }
        ]
    }

    rows = generate_rows(plan, client=_RetryClient())

    assert rows[0]["question"] == "What is the total OEM demand by region and quarter?"
    assert rows[0]["wording_warnings"] == []


def test_generate_rows_retries_when_llm_returns_malformed_json():
    plan = {
        "rows": [
            {
                "template_id": "t1",
                "family": "autonomous_driving",
                "answer_shape": "ranking_top",
                "target_ambiguity_label": "low",
                "seed_ambiguity_label": "low",
                "source_id": "S1",
                "example_question": "Which vehicle type has the highest percentage at Level 5 autonomy?",
                "query": "SELECT ...",
            }
        ]
    }

    rows = generate_rows(plan, client=_MalformedThenValidClient())

    assert rows[0]["question"] == "Which vehicle type has the highest percentage at Level 5 autonomy?"


def test_generate_rows_calls_progress_hook_after_each_row():
    plan = {
        "rows": [
            {
                "template_id": "t1",
                "family": "regional_demand",
                "answer_shape": "sum",
                "target_ambiguity_label": "high",
                "seed_ambiguity_label": "low",
                "source_id": "S1",
                "example_question": "Return total demand by region.",
                "query": "SELECT ...",
            }
        ]
    }
    snapshots = []

    generate_rows(plan, client=_FakeClient(), on_row=lambda rows: snapshots.append(list(rows)))

    assert len(snapshots) == 1
    assert snapshots[0][0]["id"] == "FINALKGQA001"
