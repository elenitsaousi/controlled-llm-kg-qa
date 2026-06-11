from kg.capabilities import DEFAULT_REGISTRY


def _names(items):
    return {item.name for item in items}


def test_future_demand_typo_resolves_to_capability_and_region_requirements():
    report = DEFAULT_REGISTRY.resolve("how does futurer demand change by region")

    assert report.primary_capability == "future demand"
    assert "region" in _names(report.detected_dimensions)
    required = dict(report.required_terms)
    assert "capability:future demand" in required
    assert "FutureDemandAnalysis" in required["capability:future demand"]
    assert "dimension:region" in required
    assert any("possible typo" in warning for warning in report.typo_warnings)


def test_dimension_typos_resolve_generically():
    cases = [
        ("demand by vehcle type", "vehicle type"),
        ("demand by technolgy category", "technology category"),
        ("demand by quater", "quarter"),
    ]
    for question, expected_dimension in cases:
        report = DEFAULT_REGISTRY.resolve(question)
        assert expected_dimension in _names(report.detected_dimensions)


def test_regional_demand_plural_alias_resolves():
    report = DEFAULT_REGISTRY.resolve("regional demands by vehicle type")

    assert report.primary_capability == "regional demand"
    assert "vehicle type" in _names(report.detected_dimensions)


def test_semantic_coverage_not_perfect_when_core_capability_missing():
    report = DEFAULT_REGISTRY.evaluate_query(
        "how does futurer demand change by region",
        """
        SELECT ?region (AVG(?percentageChange) AS ?avgChange) WHERE {
          ?row survey:forRegion ?region ;
               survey:percentageChange ?percentageChange .
        }
        GROUP BY ?region
        """,
    )

    assert report.coverage_score < 1.0
    assert "capability:future demand" in report.missing_required_terms
    assert "dimension:region" in report.covered_required_terms


def test_capability_suggestions_are_dimension_specific():
    suggestions = DEFAULT_REGISTRY.capability_suggestions("future demand")
    labels = {str(row["label"]).lower() for row in suggestions}

    assert any("region" in label for label in labels)
    assert any("quarter" in label for label in labels)
    assert not any("month" in label for label in labels)


def test_specific_future_demand_region_has_direct_query():
    report = DEFAULT_REGISTRY.resolve("how does Future demand change by region")
    query = DEFAULT_REGISTRY.direct_query_for(report)

    assert report.primary_capability == "future demand"
    assert [dimension.name for dimension in report.detected_dimensions] == ["region"]
    assert query is not None
    assert "DemandForRegion" in query
    assert "GROUP BY ?regionName" in query
    assert "COUNT(" not in query.upper()


def test_future_demand_by_vehicle_type_defaults_to_grouped_breakdown():
    report = DEFAULT_REGISTRY.resolve("future demand by vehicle type")
    query = DEFAULT_REGISTRY.direct_query_for(report)

    assert report.primary_capability == "future demand"
    assert [dimension.name for dimension in report.detected_dimensions] == ["vehicle type"]
    assert query is not None
    assert "GROUP BY ?vehicleType" in query
    assert "LIMIT 1" not in query.upper()
    assert "ORDER BY DESC" not in query.upper()
