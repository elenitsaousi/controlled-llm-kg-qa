from evaluation.generate_final_kgqa_benchmark_llm import _parse_question, generate_rows


class _FakeClient:
    def generate_text(self, _prompt):
        return '{"question":"How does demand vary by region?"}'


def test_parse_question_accepts_json_object():
    assert _parse_question('{"question":"What is demand"}') == "What is demand?"


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
