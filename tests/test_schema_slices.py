from kg.schema import KGSchema
from kg.schema_slices import build_schema_slice, infer_schema_slice_names


def _schema() -> KGSchema:
    return KGSchema(
        {
            "description": "test",
            "classes": [
                "Company",
                "Region",
                "Quarter",
                "TechnologyCategory",
                "InventoryDevelopment_Tier1",
                "Component",
                "FutureDemandAnalysis",
                "VehicleSalesObservation",
            ],
            "predicates": [
                "forCompany",
                "forComponent",
                "forTimePeriod",
                "hasSurveyOrigin",
                "analyzesTechnologyCategory",
            ],
            "properties": ["inventoryTrend", "totalDemand", "companyName"],
            "relationships": [
                {
                    "type": "forComponent",
                    "from": ["InventoryDevelopment_Tier1"],
                    "to": ["Component"],
                },
                {
                    "type": "forTimePeriod",
                    "from": ["FutureDemandAnalysis", "VehicleSalesObservation"],
                    "to": ["Quarter"],
                },
            ],
        }
    )


def test_infer_schema_slice_names_from_question() -> None:
    assert infer_schema_slice_names("Show inventory by component") == ["inventory"]
    assert infer_schema_slice_names("Which month has forecast vehicle sales?") == [
        "future_demand",
        "vehicle_sales",
    ]


def test_build_schema_slice_keeps_common_context_and_relevant_relationships() -> None:
    sliced = build_schema_slice(_schema(), ["inventory"])

    assert "Company" in sliced.classes
    assert "InventoryDevelopment_Tier1" in sliced.classes
    assert "Component" in sliced.classes
    assert "forComponent" in sliced.predicates
    assert any(rel["type"] == "forComponent" for rel in sliced.relationships)
