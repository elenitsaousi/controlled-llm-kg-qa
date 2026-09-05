from evaluation.repair_generated_benchmark_rows import repair_rows


class _FakeClient:
    def generate_text(self, _prompt):
        return '{"question":"Show inventory trends by component?"}'


def test_repair_rows_replaces_only_requested_id():
    plan = {
        "rows": [
            {
                "template_id": "t1",
                "family": "inventory",
                "answer_shape": "raw_or_lookup",
                "target_ambiguity_label": "mid",
                "seed_ambiguity_label": "mid",
                "source_id": "S1",
                "example_question": "Show inventory trends by component.",
                "query": "SELECT ...",
            },
            {
                "template_id": "t2",
                "family": "regional_demand",
                "answer_shape": "sum",
                "target_ambiguity_label": "low",
                "seed_ambiguity_label": "low",
                "source_id": "S2",
                "example_question": "Return total demand by region.",
                "query": "SELECT 2 ...",
            },
        ]
    }
    rows = [
        {"id": "FINALKGQA001", "template_id": "t1", "question": "How many inventory records?", "query": "SELECT ..."},
        {"id": "FINALKGQA002", "template_id": "t2", "question": "Return total demand by region?", "query": "SELECT 2 ..."},
    ]

    repaired = repair_rows(plan, rows, ["FINALKGQA001"], client=_FakeClient())

    assert repaired[0]["question"] == "Show inventory trends by component?"
    assert repaired[1] == rows[1]
