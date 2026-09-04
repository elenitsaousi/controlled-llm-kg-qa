import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("tier 1", "tier1").replace("tier-1", "tier1")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = []
    for token in text.split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def _token_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    compact_ratio = SequenceMatcher(None, _compact(left), _compact(right)).ratio()
    return max(overlap, ratio, compact_ratio)


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    aliases: Tuple[str, ...]
    required_terms: Tuple[str, ...]
    distinct_values: Optional[int] = None
    estimated_rows: Optional[int] = None


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    aliases: Tuple[str, ...]
    core_terms: Tuple[str, ...]
    dimensions: Tuple[DimensionSpec, ...]
    aggregations: Tuple[str, ...] = ("SUM", "AVG", "COUNT")
    related_predicates: Tuple[str, ...] = ()
    query_templates: Dict[str, str] = field(default_factory=dict)

    def all_aliases(self) -> Tuple[str, ...]:
        return (self.name, *self.aliases)

    def dimension_by_name(self, name: str) -> Optional[DimensionSpec]:
        needle = normalize_text(name)
        for dimension in self.dimensions:
            values = (dimension.name, *dimension.aliases)
            if any(normalize_text(v) == needle for v in values):
                return dimension
        return None


@dataclass
class ResolvedConcept:
    kind: str
    name: str
    matched_text: str
    score: float
    exact: bool = False


@dataclass
class ResolutionReport:
    original_question: str
    canonical_question: str
    detected_capabilities: List[ResolvedConcept] = field(default_factory=list)
    detected_dimensions: List[ResolvedConcept] = field(default_factory=list)
    detected_aggregation: Optional[str] = None
    required_terms: List[Tuple[str, Tuple[str, ...]]] = field(default_factory=list)
    covered_required_terms: List[str] = field(default_factory=list)
    missing_required_terms: List[str] = field(default_factory=list)
    typo_warnings: List[str] = field(default_factory=list)

    @property
    def primary_capability(self) -> Optional[str]:
        if not self.detected_capabilities:
            return None
        return self.detected_capabilities[0].name

    @property
    def coverage_score(self) -> float:
        if not self.required_terms:
            return 0.0
        return len(self.covered_required_terms) / max(1, len(self.required_terms))


def _dim(
    name: str,
    aliases: Sequence[str],
    terms: Sequence[str],
    *,
    distinct_values: Optional[int] = None,
    estimated_rows: Optional[int] = None,
) -> DimensionSpec:
    return DimensionSpec(
        name=name,
        aliases=tuple(aliases),
        required_terms=tuple(terms),
        distinct_values=distinct_values,
        estimated_rows=estimated_rows,
    )


DEFAULT_CAPABILITIES: Tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        name="future demand",
        aliases=("future demand analysis", "future demand change", "future demand percentage", "demand forecast"),
        core_terms=("FutureDemandAnalysis",),
        dimensions=(
            _dim("region", ("regions", "regional"), ("DemandForRegion", "Region", "percentageChange", "totalDemandPercentageChange"), distinct_values=5),
            _dim("quarter", ("quarters", "quater", "quarterly", "time period"), ("Quarter", "periodLabel", "percentageChange", "totalDemandPercentageChange"), distinct_values=8),
            _dim("vehicle type", ("vehicle", "vehcle type", "vehicle category"), ("VehicleType", "percentageChange"), distinct_values=3),
            _dim("technology category", ("technology", "technolgy category", "tech category", "node"), ("TechnologyCategory", "percentageChange"), distinct_values=5),
            _dim("survey group", ("survey", "survey type", "origin", "survey origin"), ("hasSurveyOrigin",), distinct_values=3),
        ),
        aggregations=("AVG", "SUM", "COUNT", "MAX", "MIN"),
        query_templates={
            "region": """
SELECT ?regionName (AVG(?pct) AS ?avgPercentageChange) WHERE {
  ?entry a survey:DemandForRegion ;
         survey:inRegion ?region ;
         survey:totalDemandPercentageChange ?pct .
  ?region survey:regionName ?regionName .
}
GROUP BY ?regionName
ORDER BY ?regionName
""",
            "quarter": """
SELECT ?quarterLabel (AVG(?pct) AS ?avgPercentageChange) WHERE {
  ?entry a survey:FutureDemandAnalysis ;
         survey:forTimePeriod ?quarter ;
         survey:percentageChange ?pct .
  ?quarter survey:periodLabel ?quarterLabel .
}
GROUP BY ?quarterLabel
ORDER BY ?quarterLabel
""",
            "vehicle type": """
SELECT ?vehicleType (AVG(?pct) AS ?avgPercentageChange) WHERE {
  ?entry a survey:FutureDemandAnalysis ;
         survey:analyzesVehicleType ?vehicle ;
         survey:percentageChange ?pct .
  BIND(REPLACE(STR(?vehicle), "^.*/", "") AS ?vehicleType)
}
GROUP BY ?vehicleType
ORDER BY ?vehicleType
""",
            "technology category": """
SELECT ?technologyCategory (AVG(?pct) AS ?avgPercentageChange) WHERE {
  ?entry a survey:FutureDemandAnalysis ;
         survey:analyzesTechnologyCategory ?technology ;
         survey:percentageChange ?pct .
  OPTIONAL { ?technology survey:technologyCategoryName ?technologyName . }
  BIND(COALESCE(?technologyName, REPLACE(STR(?technology), "^.*/", "")) AS ?technologyCategory)
}
GROUP BY ?technologyCategory
ORDER BY ?technologyCategory
""",
            "survey group": """
SELECT ?surveyGroup (AVG(?pct) AS ?avgPercentageChange) WHERE {
  ?entry a survey:DemandForSurveyGroup ;
         survey:hasSurveyOrigin ?origin ;
         survey:percentageChange ?pct .
  BIND(REPLACE(STR(?origin), "^.*/", "") AS ?surveyGroup)
}
GROUP BY ?surveyGroup
ORDER BY ?surveyGroup
""",
        },
    ),
    CapabilitySpec(
        name="regional demand",
        aliases=("current regional demand", "demand by region", "regional demands", "current demand"),
        core_terms=("CurrentDemandAnalysis", "AggregatedDemand", "DemandForRegion"),
        dimensions=(
            _dim("region", ("regions", "regional"), ("DemandForRegion", "Region", "regionName")),
            _dim("quarter", ("quarters", "quater"), ("Quarter", "periodLabel")),
            _dim("survey group", ("survey", "survey type", "origin"), ("hasSurveyOrigin", "OEM_Survey", "Tier1_Survey", "Semiconductor_Survey")),
            _dim("vehicle type", ("vehicle", "vehcle type"), ("VehicleType", "hasVehicleType")),
        ),
        aggregations=("SUM", "AVG", "COUNT", "MAX", "MIN"),
    ),
    CapabilitySpec(
        name="vehicle sales",
        aliases=("actual vehicle sales", "forecast vehicle sales", "vehicle units sold", "vehicles sold", "sales units", "sales volume"),
        core_terms=("VehicleSalesObservation",),
        dimensions=(
            _dim("month", ("months", "monthly", "time period", "time periods"), ("periodLabel", "Month"), distinct_values=12),
            _dim("year", ("years", "yearly"), ("hasYear",)),
            _dim("vehicle type", ("vehicle", "vehcle type"), ("VehicleType", "hasVehicleType")),
        ),
        aggregations=("SUM", "AVG", "COUNT"),
    ),
    CapabilitySpec(
        name="shortage",
        aliases=("shortages", "shortage status", "companies reporting shortage", "reported shortage"),
        core_terms=("Company", "reportsShortage"),
        dimensions=(
            _dim("shortage status", ("shortage", "reported shortage", "with and without shortage"), ("reportsShortage",)),
            _dim("survey group", ("survey", "survey type", "origin"), ("hasSurveyOrigin", "OEM_Survey", "Tier1_Survey", "Semiconductor_Survey")),
            _dim("technology category", ("technology", "technolgy category"), ("TechnologyCategory", "technologyCategoryName")),
        ),
        aggregations=("COUNT",),
    ),
    CapabilitySpec(
        name="autonomous driving",
        aliases=("autonomous driving development", "autonomousdrivingdevelopment", "sae level", "self driving"),
        core_terms=("AutonomousDrivingDevelopment",),
        dimensions=(
            _dim("vehicle type", ("vehicle", "vehcle type"), ("VehicleType", "hasVehicleType")),
            _dim("SAE level", ("sae", "level 5", "sae level 5"), ("SAELevel", "hasSAELevel"), distinct_values=6),
            _dim("year", ("years", "yearly"), ("hasYear",)),
            _dim("survey group", ("survey", "survey type", "origin"), ("hasSurveyOrigin", "OEM_Survey", "Tier1_Survey")),
        ),
        aggregations=("AVG", "COUNT", "MAX"),
    ),
    CapabilitySpec(
        name="inventory",
        aliases=("inventory trend", "inventory trends", "inventory entries", "inventory amount"),
        core_terms=("InventoryDevelopment",),
        dimensions=(
            _dim("component", ("components", "component type"), ("forComponent",)),
            _dim("technology category", ("technology", "technolgy category"), ("TechnologyCategory", "technologyCategoryName")),
            _dim("trend", ("inventory trend", "increase decrease stable"), ("inventoryTrend", "hasInventoryTrend")),
        ),
        aggregations=("SUM", "COUNT"),
    ),
    CapabilitySpec(
        name="order cancellation",
        aliases=("order cancellations", "order cancellation response", "cancellation response"),
        core_terms=("OrderCancellation",),
        dimensions=(
            _dim("technology category", ("technology", "technolgy category"), ("TechnologyCategory", "technologyCategoryName")),
            _dim("response type", ("response", "response trend", "increase decrease stable"), ("responseType", "hasResponseType")),
        ),
        aggregations=("SUM", "COUNT"),
    ),
    CapabilitySpec(
        name="catalog lookup",
        aliases=(
            "available names",
            "list names",
            "names of all",
            "what are the names",
            "region names",
            "technology category names",
            "quarter labels",
            "company names",
            "catalog",
            "lookup",
        ),
        core_terms=("Company", "Region", "TechnologyCategory", "Quarter"),
        dimensions=(
            _dim("companies", ("company", "firms"), ("Company", "companyName")),
            _dim("regions", ("region", "regional"), ("Region", "regionName")),
            _dim("technology categories", ("technology category", "technolgy category"), ("TechnologyCategory", "technologyCategoryName")),
            _dim("quarter labels", ("quarter", "quater"), ("Quarter", "periodLabel")),
        ),
        aggregations=("COUNT",),
    ),
)


class CapabilityRegistry:
    def __init__(self, capabilities: Iterable[CapabilitySpec] = DEFAULT_CAPABILITIES):
        self.capabilities = list(capabilities)

    def find_capability(self, name: str) -> Optional[CapabilitySpec]:
        needle = normalize_text(name)
        for capability in self.capabilities:
            if normalize_text(capability.name) == needle:
                return capability
        return None

    def resolve(self, question: str, *, min_exact_score: float = 0.74, min_typo_score: float = 0.84) -> ResolutionReport:
        q_norm = normalize_text(question)
        report = ResolutionReport(
            original_question=str(question or ""),
            canonical_question=str(question or ""),
            detected_aggregation=_detect_aggregation(q_norm),
        )
        capability_matches: List[ResolvedConcept] = []
        for capability in self.capabilities:
            if capability.name == "catalog lookup" and not _has_catalog_lookup_intent(q_norm):
                continue
            match = _best_phrase_match(q_norm, capability.all_aliases())
            if not match:
                continue
            matched_text, score, exact = match
            if exact or score >= min_exact_score:
                capability_matches.append(ResolvedConcept("capability", capability.name, matched_text, score, exact))
                if not exact and score >= min_typo_score:
                    report.typo_warnings.append(f"possible typo: '{matched_text}' -> '{capability.name}'")
        capability_matches.sort(key=lambda item: (item.exact, item.score, len(item.name)), reverse=True)
        if re.search(r"\b(future demand|future demand analysis|future demand change|demand forecast)\b", q_norm):
            capability_matches.sort(
                key=lambda item: (
                    item.name == "future demand",
                    item.exact,
                    item.score,
                    len(item.name),
                ),
                reverse=True,
            )
        report.detected_capabilities = capability_matches[:3]

        active_capabilities = [
            self.find_capability(item.name)
            for item in report.detected_capabilities
            if self.find_capability(item.name)
        ] or self.capabilities

        dimension_matches: List[ResolvedConcept] = []
        seen_dims = set()
        for capability in active_capabilities:
            for dimension in capability.dimensions:
                match = _best_phrase_match(q_norm, (dimension.name, *dimension.aliases))
                if not match:
                    continue
                matched_text, score, exact = match
                if not (exact or score >= min_exact_score):
                    continue
                key = normalize_text(dimension.name)
                if key in seen_dims:
                    continue
                seen_dims.add(key)
                dimension_matches.append(ResolvedConcept("dimension", dimension.name, matched_text, score, exact))
                if not exact and score >= min_typo_score:
                    report.typo_warnings.append(f"possible typo: '{matched_text}' -> '{dimension.name}'")
        dimension_matches.sort(key=lambda item: (item.exact, item.score, len(item.name)), reverse=True)
        report.detected_dimensions = dimension_matches
        report.canonical_question = _canonicalize_question(str(question or ""), report)
        report.required_terms = self.required_terms(report)
        return report

    def required_terms(self, report: ResolutionReport) -> List[Tuple[str, Tuple[str, ...]]]:
        required: List[Tuple[str, Tuple[str, ...]]] = []
        for resolved in report.detected_capabilities[:1]:
            capability = self.find_capability(resolved.name)
            if not capability:
                continue
            required.append((f"capability:{capability.name}", capability.core_terms))
            for dimension_resolved in report.detected_dimensions:
                dimension = capability.dimension_by_name(dimension_resolved.name)
                if dimension:
                    required.append((f"dimension:{dimension.name}", dimension.required_terms))
        return _dedupe_required(required)

    def evaluate_query(self, question: str, query: str) -> ResolutionReport:
        report = self.resolve(question)
        normalized_query = _compact(query)
        loose_query = normalize_text(query).replace(" ", "")
        covered = []
        missing = []
        for name, terms in report.required_terms:
            term_matches = [_compact(term) for term in terms if term]
            if any(term and (term in normalized_query or term in loose_query) for term in term_matches):
                covered.append(name)
            else:
                missing.append(name)
        report.covered_required_terms = covered
        report.missing_required_terms = missing
        return report

    def capability_suggestions(self, capability_name: str) -> List[Dict[str, object]]:
        capability = self.find_capability(capability_name)
        if not capability:
            return []
        suggestions = []
        for dimension in capability.dimensions:
            suggestions.append(
                {
                    "capability": capability.name,
                    "dimension": dimension.name,
                    "label": _suggestion_label(capability.name, dimension.name),
                    "required_terms": list(capability.core_terms + dimension.required_terms),
                    "aggregations": list(capability.aggregations),
                    "distinct_values": dimension.distinct_values,
                    "estimated_rows": dimension.estimated_rows,
                }
            )
        return suggestions

    def direct_query_for(self, report: ResolutionReport) -> Optional[str]:
        if not report.primary_capability or not report.detected_capabilities:
            return None
        primary = report.detected_capabilities[0]
        if not primary.exact and primary.score < 0.9:
            return None
        if not report.detected_dimensions:
            return None
        capability = self.find_capability(report.primary_capability)
        if not capability:
            return None
        relevant_dimensions = _dimensions_for_capability(report, capability)
        relevant_dimension_names = {normalize_text(item.name) for item in relevant_dimensions}
        q_norm = normalize_text(report.original_question)
        if _direct_contract_blocks(capability.name, relevant_dimension_names, q_norm):
            return None
        dynamic_query = _direct_dynamic_query(report)
        if dynamic_query:
            return dynamic_query.strip()
        if len(relevant_dimensions) != 1:
            return None
        dimension_name = normalize_text(relevant_dimensions[0].name)
        for key, query in capability.query_templates.items():
            if normalize_text(key) == dimension_name:
                direct_query = query.strip()
                if _asks_for_rank(q_norm):
                    return _rank_query(direct_query, "?avgPercentageChange").strip()
                return direct_query
        return None


def _detect_aggregation(q_norm: str) -> Optional[str]:
    if re.search(r"\b(avg|average|mean)\b", q_norm):
        return "AVG"
    if re.search(r"\b(total|sum|aggregate|overall)\b", q_norm):
        return "SUM"
    if re.search(r"\b(count|how many|number)\b", q_norm):
        return "COUNT"
    if re.search(r"\b(highest|top|max|largest)\b", q_norm):
        return "MAX"
    if re.search(r"\b(lowest|min|smallest)\b", q_norm):
        return "MIN"
    return None


def _has_catalog_lookup_intent(q_norm: str) -> bool:
    return bool(
        re.search(
            r"\b(list|show|available|names?|labels?|catalog|lookup|what are|which are)\b",
            q_norm,
        )
        and re.search(r"\b(name|names|label|labels|available|catalog|lookup|regions?|companies|quarters?)\b", q_norm)
    )


def _best_phrase_match(q_norm: str, aliases: Sequence[str]) -> Optional[Tuple[str, float, bool]]:
    if not q_norm:
        return None
    best: Optional[Tuple[str, float, bool]] = None
    q_tokens = q_norm.split()
    for alias in aliases:
        alias_norm = normalize_text(alias)
        if not alias_norm:
            continue
        if re.search(r"\b" + re.escape(alias_norm) + r"\b", q_norm):
            candidate = (alias_norm, 1.0, True)
        else:
            alias_len = max(1, len(alias_norm.split()))
            windows = [" ".join(q_tokens[i : i + alias_len]) for i in range(0, max(1, len(q_tokens) - alias_len + 1))]
            if alias_len > 1:
                windows.extend(" ".join(q_tokens[i : i + alias_len + 1]) for i in range(0, max(1, len(q_tokens) - alias_len)))
                windows.extend(" ".join(q_tokens[i : i + alias_len - 1]) for i in range(0, max(1, len(q_tokens) - alias_len + 2)))
            scored = [(window, _token_similarity(window, alias_norm)) for window in windows if window]
            if not scored:
                continue
            window, score = max(scored, key=lambda item: item[1])
            candidate = (window, score, False)
        if best is None or candidate[1] > best[1]:
            best = candidate
    if best and best[1] >= 0.72:
        return best
    return None


def _canonicalize_question(question: str, report: ResolutionReport) -> str:
    canonical = str(question or "")
    replacements = [
        (item.matched_text, item.name)
        for item in [*report.detected_capabilities, *report.detected_dimensions]
        if not item.exact and item.score >= 0.84
    ]
    for matched, target in replacements:
        if not matched:
            continue
        canonical = re.sub(re.escape(matched), target, canonical, flags=re.I)
    return canonical


def _dedupe_required(required: List[Tuple[str, Tuple[str, ...]]]) -> List[Tuple[str, Tuple[str, ...]]]:
    seen = set()
    out = []
    for name, terms in required:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, tuple(terms)))
    return out


def _suggestion_label(capability_name: str, dimension_name: str) -> str:
    if "demand" in capability_name:
        return f"{capability_name.title()} by {dimension_name}"
    if capability_name == "shortage":
        return f"Shortage counts by {dimension_name}"
    if capability_name == "vehicle sales":
        return f"Vehicle sales by {dimension_name}"
    if capability_name == "autonomous driving":
        return f"Autonomous driving by {dimension_name}"
    return f"{capability_name.title()} by {dimension_name}"


DEFAULT_REGISTRY = CapabilityRegistry()


def _dimensions_for_capability(report: ResolutionReport, capability: CapabilitySpec) -> Tuple[ResolvedConcept, ...]:
    return tuple(
        item
        for item in report.detected_dimensions
        if capability.dimension_by_name(str(item.name))
    )


def _detected_dimension_names(report: ResolutionReport, capability: Optional[CapabilitySpec] = None) -> Tuple[str, ...]:
    dimensions = _dimensions_for_capability(report, capability) if capability else tuple(report.detected_dimensions)
    return tuple(str(item.name).strip().lower() for item in dimensions)


def _scope_from_question(question: str) -> Optional[Tuple[str, str, str]]:
    q = normalize_text(question)
    scopes = []
    if "oem" in q:
        scopes.append(("OEM", "OEM_Survey", "OEM_Survey_Instance"))
    if "tier1" in q:
        scopes.append(("Tier1", "Tier1_Survey", "Tier1_Survey_Instance"))
    semiconductor_as_survey_scope = bool(
        ("semiconductor survey" in q)
        or ("semiconductor data" in q)
        or ("semiconductor dataset" in q)
        or ("semiconductor finding" in q)
        or ("semiconductor response" in q)
        or ("semiconductor survey response" in q)
    )
    if semiconductor_as_survey_scope:
        scopes.append(("Semiconductor", "Semiconductor_Survey", "Semiconductor_Survey_Instance"))
    if len(scopes) != 1:
        return None
    return scopes[0]


def _asks_for_rank(q_norm: str) -> bool:
    return bool(re.search(r"\b(highest|top|largest|maximum|max|greatest|leads?|leading)\b", q_norm))


def _asks_for_company_list(q_norm: str) -> bool:
    return bool(
        re.search(r"\b(which|list|name|show|identify)\s+(?:the\s+)?(?:oem\s+|tier1\s+|semiconductor\s+)?(?:companies|company)\b", q_norm)
        or re.search(r"\b(?:companies|company)\s+(?:that|who|which|with)\b", q_norm)
    )


def _asks_for_count(q_norm: str) -> bool:
    return bool(re.search(r"\b(count|how many|number of|total number|total count)\b", q_norm))


def _mentions_current_demand_baseline(q_norm: str) -> bool:
    return bool(
        re.search(r"\b(bl1|bl2|baseline|option1|option2|option3|b1|b2)\b", q_norm)
        or "body and convenience" in q_norm
        or "automotive current demand" in q_norm
    )


def _mentions_actual_and_forecast(q_norm: str) -> bool:
    return bool(
        re.search(r"\b(actual|actuals|result|results)\b", q_norm)
        and re.search(r"\b(forecast|forecasted|forecasts)\b", q_norm)
    )


def _asks_for_advisory_answer(q_norm: str) -> bool:
    return bool(
        re.search(r"\b(should|recommend|advice|advise|monitor|inspect|focus|attention|risk|exposure|uncertain)\b", q_norm)
    )


def _rank_query(query: str, value_var: str) -> str:
    body = str(query or "").strip()
    body = re.sub(r"\nORDER BY[^\n]*(?:\n|$)", "\n", body, flags=re.I)
    return f"{body}\nORDER BY DESC({value_var})\nLIMIT 1"


def _direct_contract_blocks(capability: str, dims: set, q_norm: str) -> bool:
    if _asks_for_advisory_answer(q_norm):
        return True
    if _mentions_current_demand_baseline(q_norm):
        return True
    if "survey group" in dims and _asks_for_rank(q_norm):
        # Ranking by survey provenance is not represented by a stable direct template.
        return True
    if _asks_for_company_list(q_norm) and capability != "shortage":
        return True
    return False


def _survey_group_projection(scope: Optional[Tuple[str, str, str]]) -> Tuple[str, str, str]:
    if scope:
        label, _, instance = scope
        return (
            f'BIND("{label}" AS ?surveyGroup)',
            f"FILTER(?origin = survey:{instance})",
            "?surveyGroup",
        )
    return (
        """
VALUES (?origin ?surveyGroup) {
  (survey:OEM_Survey_Instance "OEM")
  (survey:Tier1_Survey_Instance "Tier1")
  (survey:Semiconductor_Survey_Instance "Semiconductor")
}
""".strip(),
        "",
        "?surveyGroup",
    )


def _direct_dynamic_query(report: ResolutionReport) -> Optional[str]:
    capability = str(report.primary_capability or "").strip().lower()
    capability_spec = DEFAULT_REGISTRY.find_capability(capability)
    dims = set(_detected_dimension_names(report, capability_spec)) if capability_spec else set(_detected_dimension_names(report))
    q_norm = normalize_text(report.original_question)
    if _direct_contract_blocks(capability, dims, q_norm):
        return None

    if capability == "regional demand":
        return _regional_demand_direct_query(dims, q_norm)
    if capability == "vehicle sales":
        return _vehicle_sales_direct_query(dims, q_norm)
    if capability == "shortage":
        return _shortage_direct_query(dims, q_norm)
    if capability == "autonomous driving":
        return _autonomous_driving_direct_query(dims, q_norm)
    if capability == "inventory":
        return _inventory_direct_query(dims, q_norm)
    if capability == "order cancellation":
        return _order_cancellation_direct_query(dims, q_norm)
    if capability == "catalog lookup":
        return _catalog_lookup_direct_query(dims, q_norm)
    return None


def _has_any(q_norm: str, phrases: Sequence[str]) -> bool:
    return any(phrase in q_norm for phrase in phrases)


def _regional_demand_direct_query(dims: set, q_norm: str) -> Optional[str]:
    scope = _scope_from_question(q_norm)
    bind_survey, origin_type, survey_var = _survey_group_projection(scope)
    quarter_requested = _has_any(
        q_norm,
        (
            "by quarter",
            "per quarter",
            "across quarter",
            "grouped by quarter",
            "for each quarter",
        ),
    )
    region_requested = _has_any(
        q_norm,
        (
            "by region",
            "per region",
            "across region",
            "grouped by region",
            "for each region",
            "current demand by region",
        ),
    )
    if "quarter" in dims and quarter_requested and not region_requested:
        return f"""
SELECT {'' if scope else '?surveyGroup '}?quarterLabel (SUM(?demand) AS ?totalDemand) WHERE {{
  {bind_survey}
  ?entry a survey:DemandForRegion ;
         survey:hasSurveyOrigin ?origin ;
         survey:quarter ?quarter ;
         survey:totalDemandPercentageChange ?demand .
  {origin_type}
  OPTIONAL {{ ?quarter survey:periodLabel ?periodLabel . }}
  BIND(COALESCE(?periodLabel, REPLACE(STR(?quarter), "^.*/", "")) AS ?quarterLabel)
}}
GROUP BY {'' if scope else '?surveyGroup '}?quarterLabel
ORDER BY {'' if scope else '?surveyGroup '}?quarterLabel
"""
    if dims == {"vehicle type"}:
        if scope or _has_any(q_norm, ("total", "sum", "units", "volume", "regional demand")):
            return None
        return """
SELECT ?vehicleType (AVG(?pct) AS ?avgPercentageChange) WHERE {
  ?entry a survey:CurrentDemandAnalysis ;
         survey:analyzesVehicleType ?vehicle ;
         survey:percentageChange ?pct .
  BIND(REPLACE(STR(?vehicle), "^.*/", "") AS ?vehicleType)
}
GROUP BY ?vehicleType
ORDER BY ?vehicleType
"""
    if "region" in dims:
        if "vehicle type" in dims and not region_requested:
            return None
        select_survey = "" if scope else "?surveyGroup "
        group_survey = "" if scope else "?surveyGroup "
        query = f"""
SELECT {select_survey}?regionName (SUM(?demand) AS ?totalDemand) WHERE {{
  {bind_survey}
  ?entry a survey:DemandForRegion ;
         survey:hasSurveyOrigin ?origin ;
         survey:inRegion ?region ;
         survey:totalDemand ?demand .
  {origin_type}
  ?region survey:regionName ?regionName .
}}
GROUP BY {group_survey}?regionName
ORDER BY {group_survey}?regionName
"""
        return _rank_query(query, "?totalDemand") if _asks_for_rank(q_norm) else query
    if dims == {"survey group"}:
        query = f"""
SELECT {survey_var} (SUM(?demand) AS ?totalDemand) WHERE {{
  {bind_survey}
  ?entry a survey:DemandForRegion ;
         survey:hasSurveyOrigin ?origin ;
         survey:totalDemand ?demand .
  {origin_type}
}}
GROUP BY {survey_var}
ORDER BY {survey_var}
"""
        return _rank_query(query, "?totalDemand") if _asks_for_rank(q_norm) else query
    return None


def _vehicle_sales_direct_query(dims: set, q_norm: str) -> Optional[str]:
    explicit_vehicle_type = any(phrase in q_norm for phrase in ("vehicle type", "by type", "grouped by type", "for every vehicle type"))
    if "vehicle type" in dims:
        if explicit_vehicle_type:
            return None
        dims = set(dims)
        dims.discard("vehicle type")
    dims = set(dims) & {"month", "year"}
    if not (dims & {"month", "year"}):
        return None
    data_filter = ""
    split_actual_forecast = _mentions_actual_and_forecast(q_norm)
    if split_actual_forecast:
        data_filter = ""
    elif "actual" in q_norm:
        data_filter = "survey:isActualData true ;"
    elif "forecast" in q_norm or "forecasted" in q_norm:
        data_filter = "survey:isForecastData true ;"
    if dims == {"month"}:
        if split_actual_forecast:
            return """
SELECT ?monthLabel ?salesType (SUM(?units) AS ?unitsSold) WHERE {
  ?obs a survey:VehicleSalesObservation ;
       survey:forTimePeriod ?month ;
       survey:unitsSold ?units .
  OPTIONAL { ?month survey:periodLabel ?periodLabel . }
  BIND(COALESCE(?periodLabel, REPLACE(STR(?month), "^.*/", "")) AS ?monthLabel)
  BIND(IF(EXISTS { ?obs survey:isActualData true }, "actual", "forecast") AS ?salesType)
}
GROUP BY ?monthLabel ?salesType
ORDER BY ?monthLabel ?salesType
"""
        query = f"""
SELECT ?monthLabel (SUM(?units) AS ?unitsSold) WHERE {{
  ?obs a survey:VehicleSalesObservation ;
       {data_filter}
       survey:forTimePeriod ?month ;
       survey:unitsSold ?units .
  OPTIONAL {{ ?month survey:periodLabel ?periodLabel . }}
  BIND(COALESCE(?periodLabel, REPLACE(STR(?month), "^.*/", "")) AS ?monthLabel)
}}
GROUP BY ?monthLabel
ORDER BY ?monthLabel
"""
        return _rank_query(query, "?unitsSold") if _asks_for_rank(q_norm) else query
    if dims == {"year"}:
        if split_actual_forecast:
            return """
SELECT ?year ?salesType (SUM(?units) AS ?unitsSold) WHERE {
  ?obs a survey:VehicleSalesObservation ;
       survey:forTimePeriod ?month ;
       survey:unitsSold ?units .
  OPTIONAL { ?month survey:periodLabel ?periodLabel . }
  BIND(COALESCE(?periodLabel, REPLACE(STR(?month), "^.*/", "")) AS ?label)
  BIND(REPLACE(STR(?label), "^.*(20[0-9]{2}).*$", "$1") AS ?year)
  BIND(IF(EXISTS { ?obs survey:isActualData true }, "actual", "forecast") AS ?salesType)
}
GROUP BY ?year ?salesType
ORDER BY ?year ?salesType
"""
        query = f"""
SELECT ?year (SUM(?units) AS ?unitsSold) WHERE {{
  ?obs a survey:VehicleSalesObservation ;
       {data_filter}
       survey:forTimePeriod ?month ;
       survey:unitsSold ?units .
  OPTIONAL {{ ?month survey:periodLabel ?periodLabel . }}
  BIND(COALESCE(?periodLabel, REPLACE(STR(?month), "^.*/", "")) AS ?label)
  BIND(REPLACE(STR(?label), "^.*(20[0-9]{{2}}).*$", "$1") AS ?year)
}}
GROUP BY ?year
ORDER BY ?year
"""
        return _rank_query(query, "?unitsSold") if _asks_for_rank(q_norm) else query
    return None


def _shortage_direct_query(dims: set, q_norm: str) -> Optional[str]:
    scope = _scope_from_question(q_norm)
    bind_survey, origin_type, survey_var = _survey_group_projection(scope)
    asks_negative = bool(re.search(r"\b(not|no|without|did not|have not|has not)\b", q_norm))
    asks_positive = bool(
        asks_negative is False
        and re.search(r"\b(reported|reporting|facing|experiencing|identified|indicated|acknowledged|stated|disclosed|marked as having|with shortages?)\b", q_norm)
    )
    shortage_filter = "false" if asks_negative else "true" if asks_positive else ""
    if "survey group" in dims and shortage_filter and _asks_for_count(q_norm):
        return f"""
SELECT {survey_var} (COUNT(?company) AS ?companyCount) WHERE {{
  {bind_survey}
  ?company a survey:Company ;
           survey:hasSurveyOrigin ?origin ;
           survey:reportsShortage ?shortage .
  {origin_type}
  FILTER(?shortage = {shortage_filter})
}}
GROUP BY {survey_var}
ORDER BY {survey_var}
"""
    if shortage_filter and scope and _asks_for_count(q_norm) and not re.search(r"\b(which|list|identify|name)\b", q_norm):
        return f"""
SELECT (COUNT(?company) AS ?companyCount) WHERE {{
  {bind_survey}
  ?company a survey:Company ;
           survey:hasSurveyOrigin ?origin ;
           survey:reportsShortage ?shortage .
  {origin_type}
  FILTER(?shortage = {shortage_filter})
}}
"""
    if _asks_for_company_list(q_norm):
        filter_line = f"FILTER(?shortage = {shortage_filter})" if shortage_filter else ""
        return f"""
SELECT DISTINCT ?companyName WHERE {{
  {bind_survey}
  ?company a survey:Company ;
           survey:hasSurveyOrigin ?origin ;
           survey:reportsShortage ?shortage .
  {origin_type}
  {filter_line}
  OPTIONAL {{ ?company survey:companyName ?name . }}
  BIND(COALESCE(?name, REPLACE(STR(?company), "^.*/", "")) AS ?companyName)
}}
ORDER BY ?companyName
"""
    if dims == {"shortage status"}:
        return f"""
SELECT ?shortageStatus (COUNT(?company) AS ?companyCount) WHERE {{
  {bind_survey}
  ?company a survey:Company ;
           survey:hasSurveyOrigin ?origin ;
           survey:reportsShortage ?shortage .
  {origin_type}
  BIND(IF(?shortage = true, "yes", "no") AS ?shortageStatus)
}}
GROUP BY ?shortageStatus
ORDER BY ?shortageStatus
"""
    if dims in ({"survey group"}, {"survey group", "shortage status"}):
        query = f"""
SELECT {survey_var} ?shortageStatus (COUNT(?company) AS ?companyCount) WHERE {{
  {bind_survey}
  ?company a survey:Company ;
           survey:hasSurveyOrigin ?origin ;
           survey:reportsShortage ?shortage .
  {origin_type}
  BIND(IF(?shortage = true, "yes", "no") AS ?shortageStatus)
}}
GROUP BY {survey_var} ?shortageStatus
ORDER BY {survey_var} ?shortageStatus
"""
        return _rank_query(query, "?companyCount") if _asks_for_rank(q_norm) else query
    return None


def _autonomous_driving_direct_query(dims: set, q_norm: str) -> Optional[str]:
    supported = {"vehicle type", "sae level", "year"}
    if not dims or not dims.issubset(supported):
        return None
    if "sae level" in dims and re.search(r"\b(count|how many|number of|total count)\b", q_norm):
        return """
SELECT (COUNT(DISTINCT ?sae) AS ?saeLevelCount) WHERE {
  ?entry a survey:AutonomousDrivingDevelopment ;
         survey:hasSAELevel ?sae .
}
"""
    scope = _scope_from_question(q_norm)
    if scope and scope[1] == "Semiconductor_Survey":
        return None
    if scope:
        root_class = "AutonomousDrivingDevelopment_OEM" if scope[1] == "OEM_Survey" else "AutonomousDrivingDevelopment_Tier1"
        where_start = f"""
?root a survey:{root_class} ;
      survey:hasSurveyOrigin survey:{scope[1]} ;
      survey:hasDetail ?entry .
?entry a survey:AutonomousDrivingDevelopment ;
"""
    else:
        where_start = "?entry a survey:AutonomousDrivingDevelopment ;"

    select_parts = []
    group_parts = []
    binds = []
    triples = [where_start, "survey:hasPercentage ?pct ."]
    if "vehicle type" in dims:
        triples.insert(-1, "survey:hasVehicleType ?vehicle ;")
        select_parts.append("?vehicleType")
        group_parts.append("?vehicleType")
        binds.append('BIND(REPLACE(STR(?vehicle), "^.*/", "") AS ?vehicleType)')
    if "sae level" in dims:
        triples.insert(-1, "survey:hasSAELevel ?sae ;")
        select_parts.append("?saeLevel")
        group_parts.append("?saeLevel")
        binds.append('BIND(REPLACE(STR(?sae), "^.*/", "") AS ?saeLevel)')
    if "year" in dims:
        triples.insert(-1, "survey:hasYear ?year ;")
        select_parts.append("?year")
        group_parts.append("?year")
    query = f"""
SELECT {' '.join(select_parts)} (AVG(?pct) AS ?avgPercentage) WHERE {{
  {' '.join(triples)}
  {' '.join(binds)}
}}
GROUP BY {' '.join(group_parts)}
ORDER BY {' '.join(group_parts)}
"""
    return _rank_query(query, "?avgPercentage") if _asks_for_rank(q_norm) else query


def _inventory_direct_query(dims: set, q_norm: str) -> Optional[str]:
    if "trend" in dims and not any(phrase in q_norm for phrase in ("trend", "increase", "decrease", "stable")):
        dims = set(dims)
        dims.discard("trend")
    if dims == {"component"}:
        query = """
SELECT ?component (SUM(?participants) AS ?participantTotal) WHERE {
  ?entry a survey:InventoryDevelopment_Tier1 ;
         survey:forComponent ?component ;
         survey:participantCount ?participants .
}
GROUP BY ?component
ORDER BY ?component
"""
        return _rank_query(query, "?participantTotal") if _asks_for_rank(q_norm) else query
    if dims == {"component", "trend"}:
        return """
SELECT ?component ?trend (SUM(?participants) AS ?participantTotal) WHERE {
  ?entry a survey:InventoryDevelopment_Tier1 ;
         survey:forComponent ?component ;
         survey:inventoryTrend ?trend ;
         survey:participantCount ?participants .
}
GROUP BY ?component ?trend
ORDER BY ?component ?trend
"""
    if dims == {"technology category"}:
        query = """
SELECT ?technologyCategory (COUNT(?entry) AS ?entryCount) WHERE {
  ?entry a survey:InventoryDevelopment_Semi ;
         survey:forTechnologyCategory ?technology .
  BIND(REPLACE(STR(?technology), "^.*/", "") AS ?technologyCategory)
}
GROUP BY ?technologyCategory
ORDER BY ?technologyCategory
"""
        return _rank_query(query, "?entryCount") if _asks_for_rank(q_norm) else query
    if dims == {"technology category", "trend"}:
        return """
SELECT ?technologyCategory ?trend (COUNT(?entry) AS ?entryCount) WHERE {
  ?entry a survey:InventoryDevelopment_Semi ;
         survey:forTechnologyCategory ?technology ;
         survey:hasInventoryTrend ?trend .
  BIND(REPLACE(STR(?technology), "^.*/", "") AS ?technologyCategory)
}
GROUP BY ?technologyCategory ?trend
ORDER BY ?technologyCategory ?trend
"""
    return None


def _order_cancellation_direct_query(dims: set, q_norm: str) -> Optional[str]:
    if "response type" in dims and not any(
        phrase in q_norm
        for phrase in ("response type", "response trend", "increase", "decrease", "stable")
    ):
        dims = set(dims)
        dims.discard("response type")
    if dims == {"technology category"}:
        query = """
SELECT ?technologyCategory (SUM(?participants) AS ?participantCount) WHERE {
  ?entry a survey:OrderCancellation ;
         survey:forTechnologyCategory ?technology ;
         survey:participantCount ?participants .
  BIND(REPLACE(STR(?technology), "^.*/", "") AS ?technologyCategory)
}
GROUP BY ?technologyCategory
ORDER BY ?technologyCategory
"""
        return _rank_query(query, "?participantCount") if _asks_for_rank(q_norm) else query
    if dims == {"response type"}:
        query = """
SELECT ?responseType (SUM(?participants) AS ?participantCount) WHERE {
  ?entry a survey:OrderCancellation ;
         survey:hasResponseType ?responseType ;
         survey:participantCount ?participants .
}
GROUP BY ?responseType
ORDER BY ?responseType
"""
        return _rank_query(query, "?participantCount") if _asks_for_rank(q_norm) else query
    if dims == {"technology category", "response type"}:
        return """
SELECT ?technologyCategory ?responseType (SUM(?participants) AS ?participantCount) WHERE {
  ?entry a survey:OrderCancellation ;
         survey:forTechnologyCategory ?technology ;
         survey:hasResponseType ?responseType ;
         survey:participantCount ?participants .
  BIND(REPLACE(STR(?technology), "^.*/", "") AS ?technologyCategory)
}
GROUP BY ?technologyCategory ?responseType
ORDER BY ?technologyCategory ?responseType
"""
    return None


def _catalog_lookup_direct_query(dims: set, q_norm: str) -> Optional[str]:
    if dims == {"companies"} and re.search(r"\b(count|how many|number of|total)\b", q_norm):
        return """
SELECT (COUNT(DISTINCT ?company) AS ?companyCount) WHERE {
  ?company a survey:Company .
}
"""
    if dims == {"regions"}:
        return """
SELECT DISTINCT ?regionName WHERE {
  ?region a survey:Region ;
          survey:regionName ?regionName .
}
ORDER BY ?regionName
"""
    if dims == {"technology categories"}:
        return """
SELECT DISTINCT ?technologyCategory WHERE {
  ?technology a survey:TechnologyCategory .
  OPTIONAL { ?technology survey:technologyCategoryName ?technologyName . }
  BIND(COALESCE(?technologyName, REPLACE(STR(?technology), "^.*/", "")) AS ?technologyCategory)
}
ORDER BY ?technologyCategory
"""
    if dims == {"quarter labels"}:
        return """
SELECT DISTINCT ?quarterLabel WHERE {
  ?quarter a survey:Quarter ;
           survey:periodLabel ?quarterLabel .
}
ORDER BY ?quarterLabel
"""
    if dims == {"companies"}:
        return """
SELECT DISTINCT ?companyName WHERE {
  ?company a survey:Company ;
           survey:companyName ?companyName .
}
ORDER BY ?companyName
"""
    return None
