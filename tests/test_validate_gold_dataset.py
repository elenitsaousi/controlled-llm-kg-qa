from evaluation.validate_gold_dataset import validate_dataset


class _FakeResult(list):
    pass


class _FakeGraph:
    def parse(self, *_args, **_kwargs):
        return self

    def query(self, query):
        if "BROKEN" in query:
            raise ValueError("bad query")
        if "EMPTY" in query:
            return _FakeResult()
        return _FakeResult([("ok",)])


def test_validate_dataset_tracks_valid_and_non_empty(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        """[
          {"id":"A","question":"q1","query":"SELECT ?x WHERE { ?x ?p ?o }"},
          {"id":"B","question":"q2","query":"EMPTY"},
          {"id":"C","question":"q3","query":"BROKEN"}
        ]""",
        encoding="utf-8",
    )
    monkeypatch.setattr("evaluation.validate_gold_dataset.Graph", _FakeGraph)

    report = validate_dataset(str(dataset), "graph.ttl")

    assert report["summary"]["total"] == 3
    assert report["summary"]["valid"] == 2
    assert report["summary"]["non_empty"] == 1
