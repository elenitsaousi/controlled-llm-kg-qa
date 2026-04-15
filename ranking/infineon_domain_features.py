# ranking/infineon_domain_features.py
"""
Domain-aware feature extraction for Infineon survey KG queries.
"""

def extract_infineon_features(query: str, question: str) -> dict:
    q = query.upper()
    qq = question.lower()
    
    features = {}
    
    # 1. Uses correct named instances
    features["uses_named_instance"] = int(any(inst in query for inst in [
        "Tier1CurrentDemand", "OEMCurrentDemand", "SemiCurrentDemand",
        "SemiFutureDemand_Option1", "SemiFutureDemand_Option2", "SemiFutureDemand_Option3",
        "Tier1FutureDemand_Option1", "Tier1FutureDemand_Option2", "Tier1FutureDemand_Option3",
        "OEMFutureDemand_Option1", "OEMFutureDemand_Option2", "OEMFutureDemand_Option3",
    ]))
    
    # 2. Uses correct survey origin
    features["uses_survey_origin"] = int(any(s in query for s in [
        "hasSurveyOrigin", "OEM_Survey", "Tier1_Survey", "Semiconductor_Survey"
    ]))
    
    # 3. Uses correct baseline pattern
    features["uses_baseline"] = int(
        "baselineType" in query and
        ("BL1" in query or "BL2" in query)
    )
    
    # 4. Uses aggregation when needed
    needs_aggregation = any(w in qq for w in [
        "total", "sum", "average", "count", "how many", "per region"
    ])
    has_aggregation = any(w in q for w in ["SUM", "AVG", "COUNT", "GROUP BY"])
    features["aggregation_match"] = int(
        (needs_aggregation and has_aggregation) or
        (not needs_aggregation and not has_aggregation)
    )
    
    # 5. Uses regional pattern correctly
    features["uses_regional_pattern"] = int(
        "DemandForRegion" in query and
        "inRegion" in query and
        "regionName" in query
    )
    
    # 6. Uses shortage pattern correctly
    features["uses_shortage_pattern"] = int(
        "reportsShortage" in query and
        "Company" in query
    )
    
    # 7. Uses autonomous driving pattern correctly
    features["uses_autonomous_pattern"] = int(
        "AutonomousDrivingDevelopment" in query and
        "hasDetail" in query
    )
    
    # 8. Uses future demand pattern correctly
    features["uses_future_pattern"] = int(
        "FutureDemandAnalysis" in query and
        "forTimePeriod" in query
    )
    
    # 9. Has hasAggregatedResult (key pattern)
    features["uses_aggregated_result"] = int(
        "hasAggregatedResult" in query
    )
    
    # 10. Question-query alignment
    is_tier1 = "tier1" in qq or "tier 1" in qq
    is_oem = "oem" in qq
    is_semi = "semiconductor" in qq or "semi" in qq
    
    uses_tier1 = "Tier1" in query or "tier1" in query.lower()
    uses_oem = "OEM" in query
    uses_semi = "Semiconductor" in query or "Semi" in query
    
    features["survey_alignment"] = int(
        (is_tier1 and uses_tier1) or
        (is_oem and uses_oem) or
        (is_semi and uses_semi) or
        (not is_tier1 and not is_oem and not is_semi)
    )
    
    return features


WEIGHTS = {
    "uses_named_instance": 3.0,      # most important!
    "uses_survey_origin": 2.0,
    "uses_baseline": 2.0,
    "aggregation_match": 1.5,
    "uses_regional_pattern": 2.0,
    "uses_shortage_pattern": 2.0,
    "uses_autonomous_pattern": 2.0,
    "uses_future_pattern": 2.0,
    "uses_aggregated_result": 1.5,
    "survey_alignment": 1.0,
}


def score_infineon_query(query: str, question: str) -> float:
    features = extract_infineon_features(query, question)
    return sum(WEIGHTS[k] * v for k, v in features.items())


def rank_infineon_candidates(candidates: list, question: str) -> list:
    """Rank candidates using Infineon domain-aware scoring."""
    for c in candidates:
        query = c.get("query", c.get("query_text", ""))
        features = extract_infineon_features(query, question)
        c["score"] = sum(WEIGHTS[k] * features.get(k, 0) for k in WEIGHTS)
        c["features"] = features
    return sorted(candidates, key=lambda x: x["score"], reverse=True)