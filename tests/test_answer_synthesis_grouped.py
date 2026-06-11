from llm.answer_synthesis import synthesize_answer


def test_grouped_autonomous_driving_answer_does_not_convert_to_highest():
    answer = synthesize_answer(
        "What is the average autonomous driving development percentage by vehicle type and SAE level?",
        """
        SELECT ?vehicleType ?saeLevel (AVG(?percentage) AS ?avgPct) WHERE {
          ?entry a survey:AutonomousDrivingDevelopment ;
                 survey:hasVehicleType ?vehicleType ;
                 survey:hasSAELevel ?saeLevel ;
                 survey:hasPercentage ?percentage .
        }
        GROUP BY ?vehicleType ?saeLevel
        ORDER BY ?vehicleType ?saeLevel
        """,
        {
            "rows": [
                {"vehicleType": "BEV", "saeLevel": "3", "avgPct": "12.5"},
                {"vehicleType": "PHEV", "saeLevel": "4", "avgPct": "18.0"},
            ]
        },
        None,
    )

    assert "grouped row(s)" in answer
    assert "highest" not in answer.lower()
    assert "BEV" in answer
    assert "PHEV" in answer

