# prompts.py
from kg.schema import KGSchema
from ranking.feature_extraction import extract_question_entities, extract_question_relations


def _detect_query_pattern(question: str) -> str:
    """Detect query pattern based on question type and return targeted hint."""
    q = question.lower()

    hints = []

    # Highest/Lowest/Top/Bottom → ORDER BY + LIMIT 1
    if any(w in q for w in ["highest", "lowest", "most", "least", "top", "bottom", "best", "worst", "maximum", "minimum"]):
        hints.append("IMPORTANT: Use ORDER BY DESC(?value) LIMIT 1 to find the highest value, or ORDER BY ASC(?value) LIMIT 1 for lowest.")

    # Count → COUNT + GROUP BY
    if any(w in q for w in ["how many", "count", "number of", "total number"]):
        hints.append("IMPORTANT: Use COUNT(?x) AS ?count. Use BIND(IF(?val = true, 'yes', 'no') AS ?status) for boolean fields.")

    # Average → AVG + GROUP BY
    if any(w in q for w in ["average", "avg", "mean"]):
        hints.append("IMPORTANT: Use AVG(?x) AS ?avg with GROUP BY.")

    # Total/Sum → SUM + GROUP BY
    if any(w in q for w in ["total", "sum", "aggregate", "across all"]):
        hints.append("IMPORTANT: Use SUM(?x) AS ?total with GROUP BY.")

    # Trend/Evolution → ORDER BY time
    if any(w in q for w in ["trend", "evolve", "evolution", "over time", "per quarter", "by year"]):
        hints.append("IMPORTANT: Use ORDER BY ?timePeriod or ORDER BY ?year to show temporal trends.")

    # Compare → multiple values
    if any(w in q for w in ["compare", "difference", "vs", "versus", "between", "bl1", "bl2", "baseline"]):
        hints.append("IMPORTANT: Use SUM(IF(?baseline = 'BL1', ?pct, 0)) AS ?changeBL1 pattern for baseline comparisons.")

    # Regional → GROUP BY region
    if any(w in q for w in ["region", "regional", "geographic", "per region", "by region"]):
        hints.append("IMPORTANT: Use survey:DemandForRegion with survey:inRegion and survey:regionName. GROUP BY ?regionName.")

    # Shortage → reportsShortage
    if any(w in q for w in ["shortage", "shortages", "report shortage"]):
        hints.append("IMPORTANT: Use survey:reportsShortage ?Shortage with BIND(IF(?Shortage = true, 'yes', 'no') AS ?status).")

    # Future demand → SemiFutureDemand instances
    if any(w in q for w in ["future", "forecast", "projection", "projected"]):
        hints.append("IMPORTANT: Use survey:SemiFutureDemand_Option1/2/3 or survey:Tier1FutureDemand_Option1/2/3 instances.")

    # Current demand → named instances
    if any(w in q for w in ["current demand", "demand change", "baseline"]):
        hints.append("IMPORTANT: Use survey:Tier1CurrentDemand/OEMCurrentDemand/SemiCurrentDemand with survey:hasAggregatedResult.")

    # Autonomous driving
    if any(w in q for w in ["autonomous", "sae", "driving", "adas"]):
        hints.append("IMPORTANT: Use survey:AutonomousDrivingDevelopment_OEM/Tier1 with survey:hasDetail and survey:hasSAELevel.")

    # Inventory
    if any(w in q for w in ["inventory", "stock", "ev component", "non-ev", "component"]):
        hints.append("IMPORTANT: Use survey:InventoryDevelopment_Tier1 with survey:forComponent and survey:inventoryTrend.")

    # Order cancellation
    if any(w in q for w in ["order cancel", "cancellation", "cancel"]):
        hints.append("IMPORTANT: Use survey:OrderCancellation with survey:forTechnologyCategory and survey:participantCount.")

    return "\n".join(hints)


def build_candidate_prompt(question: str, schema: KGSchema, k: int = 5) -> str:
    # --- Hints (light guidance) ---
    schema_labels = schema.classes or schema.labels.keys()
    detected_entities = sorted(extract_question_entities(question, schema_labels))
    detected_relations = sorted(extract_question_relations(question))

    hints = []
    if detected_entities:
        hints.append("Detected entities: " + ", ".join(detected_entities))
    if detected_relations:
        hints.append("Detected relations: " + ", ".join(detected_relations))
    hints_text = "\n".join(hints)

    # --- Schema grounding ---
    schema_text = schema.as_prompt_text()

    # --- Dynamic query pattern hint ---
    pattern_hint = _detect_query_pattern(question)

    return (
        "You generate SPARQL SELECT queries for a real enterprise knowledge graph.\n\n"

        "CONTEXT:\n"
        "The knowledge graph represents survey data from the automotive and semiconductor industry.\n"
        "It includes companies, demand trends, inventory, regions, and technology development.\n\n"

        "NAMESPACE:\n"
        "PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>\n"
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n\n"

        "SCHEMA (authoritative):\n"
        f"{schema_text}\n\n"

        + (f"QUERY PATTERN HINTS:\n{pattern_hint}\n\n" if pattern_hint else "")

        + "GUIDELINES:\n"
        "- Before writing the query, identify the main entity and how it connects to others\n"
        "- Break the question into target entity, attributes, and conditions\n"
        "- Use the 'survey:' namespace for entities and relations\n"
        "- Queries MUST reflect the full meaning of the question\n"
        "- Prefer correct structure (joins, aggregation, grouping) over simplicity\n"
        "- Use only classes and predicates that appear in the schema section\n"
        "- Do not invent fields; if uncertain, choose the most conservative valid path\n"
        "- Use realistic domain concepts (companies, demand, regions, technology trends)\n\n"

        "SPARQL REQUIREMENTS:\n"
        "- Use SELECT queries\n"
        "- Use WHERE { ... }\n"
        "- Each triple ends with '.'\n"
        "- Use 'a' for rdf:type\n"
        "- Variables start with '?'\n"
        "- Use FILTER when needed\n"
        "- Use GROUP BY, ORDER BY, SUM, AVG when appropriate\n\n"

        "CANDIDATE REQUIREMENTS:\n"
        "- Candidate 1 MUST be the most likely correct query (highest precision, conservative schema path).\n"
        "- Other candidates may vary, but only with schema-valid alternatives.\n"
        "- Do NOT force aggregation/filter/join variants when the question does not require them.\n"
        "- Never aggregate textual/categorical values (e.g., status/trend strings) with SUM/AVG.\n"
        "- Keep candidates meaningfully different; avoid trivial variable-renaming duplicates.\n\n"

        f"{hints_text}\n\n"

        "FEW-SHOT EXAMPLES:\n\n"

        "Q: What is the regional demand for OEM?\n"
        "A:\n"
        "[\n"
        "  \"SELECT ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE { "
        "  ?demandForRegion a survey:DemandForRegion ; "
        "    survey:hasSurveyOrigin ?origin ; "
        "    survey:inRegion ?region ; "
        "    survey:totalDemand ?unitsSold . "
        "  ?origin a survey:OEM_Survey . "
        "  ?region a survey:Region ; "
        "    survey:regionName ?regionName . "
        "} GROUP BY ?regionName ORDER BY DESC(?totalDemand)\"\n"
        "]\n\n"

        "Q: Which region has the highest OEM demand?\n"
        "A:\n"
        "[\n"
        "  \"SELECT ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE { "
        "  ?demandForRegion a survey:DemandForRegion ; "
        "    survey:hasSurveyOrigin ?origin ; "
        "    survey:inRegion ?region ; "
        "    survey:totalDemand ?unitsSold . "
        "  ?origin a survey:OEM_Survey . "
        "  ?region a survey:Region ; "
        "    survey:regionName ?regionName . "
        "} GROUP BY ?regionName ORDER BY DESC(?totalDemand) LIMIT 1\"\n"
        "]\n\n"

        "Q: How many Tier1 companies report semiconductor shortage?\n"
        "A:\n"
        "[\n"
        "  \"SELECT ?ShortageStatus (COUNT(?Company) AS ?Count) WHERE { "
        "  ?Company a survey:Company ; "
        "    survey:hasSurveyOrigin ?origin ; "
        "    survey:reportsShortage ?Shortage . "
        "  ?origin a survey:Tier1_Survey . "
        "  BIND(IF(?Shortage = true, \\\"yes\\\", \\\"no\\\") AS ?ShortageStatus) "
        "} GROUP BY ?ShortageStatus\"\n"
        "]\n\n"

        "Q: What is the autonomous driving development trend for OEM?\n"
        "A:\n"
        "[\n"
        "  \"SELECT ?vehicle ?saeLabel ?year (AVG(?pct) AS ?avgPct) WHERE { "
        "  ?oemClass a survey:AutonomousDrivingDevelopment_OEM ; "
        "    survey:hasSurveyOrigin survey:OEM_Survey ; "
        "    survey:hasDetail ?entry . "
        "  ?entry a survey:AutonomousDrivingDevelopment ; "
        "    survey:hasVehicleType ?veh ; "
        "    survey:hasSAELevel ?sae ; "
        "    survey:hasPercentage ?pct ; "
        "    survey:hasYear ?year . "
        "  BIND(IF(CONTAINS(STR(?veh), \\\"BEV\\\"), \\\"BEV\\\", "
        "    IF(CONTAINS(STR(?veh), \\\"BEHV\\\"), \\\"BEHV\\\", "
        "      IF(CONTAINS(STR(?veh), \\\"ICE\\\"), \\\"ICE\\\", \\\"OTHER\\\"))) AS ?vehicle) "
        "  FILTER(?vehicle != \\\"OTHER\\\") "
        "  BIND(STRAFTER(STR(?sae), \\\"SAE_Level_\\\") AS ?saeLabel) "
        "} GROUP BY ?vehicle ?saeLabel ?year ORDER BY ?vehicle xsd:integer(?saeLabel) xsd:integer(?year)\"\n"
        "]\n\n"

        "Q: What is the total demand per region for Tier1 surveys?\n"
        "A:\n"
        "[\n"
        "  \"SELECT ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE { "
        "  ?demandForRegion a survey:DemandForRegion ; "
        "    survey:hasSurveyOrigin ?origin ; "
        "    survey:inRegion ?region ; "
        "    survey:totalDemand ?unitsSold . "
        "  ?origin a survey:Tier1_Survey . "
        "  ?region a survey:Region ; "
        "    survey:regionName ?regionName . "
        "} GROUP BY ?regionName ORDER BY DESC(?totalDemand)\"\n"
        "]\n\n"

        "Q: How does semiconductor future demand evolve across technology categories and quarters?\n"
        "A:\n"
        "[\n"
        "  \"SELECT ?techLabel ?quarter (SUM(IF(?baseline = \\\"Option1\\\", ?pct, 0)) AS ?Option1) "
        "(SUM(IF(?baseline = \\\"Option2\\\", ?pct, 0)) AS ?Option2) "
        "(SUM(IF(?baseline = \\\"Option3\\\", ?pct, 0)) AS ?Option3) WHERE { "
        "{ survey:SemiFutureDemand_Option1 a survey:FutureDemandAnalysis ; "
        "  survey:hasSurveyOrigin survey:Semiconductor_Survey ; "
        "  survey:hasAggregatedResult ?e1 . "
        "  ?e1 a survey:FutureDemandAnalysis ; "
        "  survey:analyzesTechnologyCategory ?tech ; "
        "  survey:forTimePeriod ?period ; "
        "  survey:percentageChange ?pct . "
        "  BIND(\\\"Option1\\\" AS ?baseline) } "
        "UNION { survey:SemiFutureDemand_Option2 a survey:FutureDemandAnalysis ; "
        "  survey:hasSurveyOrigin survey:Semiconductor_Survey ; "
        "  survey:hasAggregatedResult ?e2 . "
        "  ?e2 a survey:FutureDemandAnalysis ; "
        "  survey:analyzesTechnologyCategory ?tech ; "
        "  survey:forTimePeriod ?period ; "
        "  survey:percentageChange ?pct . "
        "  BIND(\\\"Option2\\\" AS ?baseline) } "
        "UNION { survey:SemiFutureDemand_Option3 a survey:FutureDemandAnalysis ; "
        "  survey:hasSurveyOrigin survey:Semiconductor_Survey ; "
        "  survey:hasAggregatedResult ?e3 . "
        "  ?e3 a survey:FutureDemandAnalysis ; "
        "  survey:analyzesTechnologyCategory ?tech ; "
        "  survey:forTimePeriod ?period ; "
        "  survey:percentageChange ?pct . "
        "  BIND(\\\"Option3\\\" AS ?baseline) } "
        "FILTER(STRSTARTS(STR(?tech), STR(survey:TechCategory_))) "
        "OPTIONAL { ?period survey:periodLabel ?qLabelRaw . } "
        "BIND(REPLACE(COALESCE(?qLabelRaw, STRAFTER(STR(?period), \\\"survey:\\\")), \\\"_\\\", \\\" \\\" ) AS ?quarter) "
        "BIND(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(STRAFTER(STR(?tech), \\\"TechCategory_\\\"), \\\"%3C%3D\\\", \\\"<=\\\"), "
        "\\\"_to_%3C\\\", \\\" to <\\\"), \\\"_or_greater\\\", \\\" or greater\\\"), \\\"_\\\", \\\" \\\"), "
        "\\\"lte 7nm\\\", \\\"<= 7nm\\\") AS ?techLabel) "
        "} GROUP BY ?techLabel ?quarter ORDER BY ?techLabel ?quarter\"\n"
        "]\n\n"

        "Q: How many semiconductor companies report shortage?\n"
        "A:\n"
        "[\n"
        "  \"SELECT ?ShortageStatus (COUNT(?Company) AS ?Count) WHERE { "
        "  ?Company a survey:Company ; "
        "    survey:hasSurveyOrigin ?origin ; "
        "    survey:reportsShortage ?Shortage . "
        "  ?origin a survey:Semiconductor_Survey . "
        "  BIND(IF(?Shortage = true, \\\"yes\\\", \\\"no\\\") AS ?ShortageStatus) "
        "} GROUP BY ?ShortageStatus\"\n"
        "]\n\n"

        f"Question: {question}\n\n"

        "OUTPUT FORMAT:\n"
        f"Return ONLY a JSON array of exactly {k} SPARQL queries.\n"
        "No explanations. No markdown. No numbering.\n"
        "Example format:\n"
        "[\"query1\", \"query2\", \"query3\"]\n\n"

        "Output:\n"
    )
