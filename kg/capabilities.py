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
        if len(token) > 3 and token.endswith("s"):
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
            _dim("quarter", ("quarters", "quater", "quarterly", "time period"), ("DemandForQuarter", "Quarter", "periodLabel", "percentageChange", "totalDemandPercentageChange"), distinct_values=8),
            _dim("vehicle type", ("vehicle", "vehcle type", "vehicle category"), ("DemandForVehicleType", "VehicleType", "percentageChange"), distinct_values=3),
            _dim("technology category", ("technology", "technolgy category", "tech category", "node"), ("DemandForTechnologyCategory", "TechnologyCategory", "percentageChange"), distinct_values=5),
            _dim("survey group", ("survey", "survey type", "origin", "survey origin"), ("DemandForSurveyGroup", "SurveyGroup", "hasSurveyOrigin"), distinct_values=3),
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
        aliases=("actual vehicle sales", "forecast vehicle sales", "vehicle units sold", "vehicles sold", "sales units"),
        core_terms=("VehicleSales", "VehicleSalesForecast"),
        dimensions=(
            _dim("month", ("months", "monthly"), ("monthLabel", "Month"), distinct_values=12),
            _dim("year", ("years", "yearly"), ("year", "hasYear")),
            _dim("vehicle type", ("vehicle", "vehcle type"), ("VehicleType", "hasVehicleType")),
        ),
        aggregations=("SUM", "AVG", "COUNT"),
    ),
    CapabilitySpec(
        name="shortage",
        aliases=("shortages", "shortage status", "companies reporting shortage", "reported shortage"),
        core_terms=("Company", "reportsShortage"),
        dimensions=(
            _dim("shortage status", ("shortage", "reported shortage", "with and without shortage"), ("reportsShortage", "ShortageStatus")),
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
            _dim("year", ("years", "yearly"), ("hasYear", "year")),
            _dim("survey group", ("survey", "survey type", "origin"), ("hasSurveyOrigin", "OEM_Survey", "Tier1_Survey")),
        ),
        aggregations=("AVG", "COUNT", "MAX"),
    ),
    CapabilitySpec(
        name="inventory",
        aliases=("inventory trend", "inventory trends", "inventory entries", "inventory amount"),
        core_terms=("InventoryDevelopment", "InventoryResponse"),
        dimensions=(
            _dim("component", ("components", "component type"), ("ComponentType", "forComponent")),
            _dim("technology category", ("technology", "technolgy category"), ("TechnologyCategory", "technologyCategoryName")),
            _dim("trend", ("inventory trend", "increase decrease stable"), ("inventoryTrend", "InventoryTrend")),
        ),
        aggregations=("SUM", "COUNT"),
    ),
    CapabilitySpec(
        name="order cancellation",
        aliases=("order cancellations", "order cancellation response", "cancellation response"),
        core_terms=("OrderCancellation", "OrderCancellationResponse"),
        dimensions=(
            _dim("technology category", ("technology", "technolgy category"), ("TechnologyCategory", "technologyCategoryName")),
            _dim("response type", ("response", "response trend", "increase decrease stable"), ("responseType", "hasResponseType")),
        ),
        aggregations=("SUM", "COUNT"),
    ),
    CapabilitySpec(
        name="catalog lookup",
        aliases=("available names", "list names", "catalog", "lookup"),
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
            match = _best_phrase_match(q_norm, capability.all_aliases())
            if not match:
                continue
            matched_text, score, exact = match
            if exact or score >= min_exact_score:
                capability_matches.append(ResolvedConcept("capability", capability.name, matched_text, score, exact))
                if not exact and score >= min_typo_score:
                    report.typo_warnings.append(f"possible typo: '{matched_text}' -> '{capability.name}'")
        capability_matches.sort(key=lambda item: (item.exact, item.score, len(item.name)), reverse=True)
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
        if len(report.detected_dimensions) != 1:
            return None
        capability = self.find_capability(report.primary_capability)
        if not capability:
            return None
        dimension_name = normalize_text(report.detected_dimensions[0].name)
        for key, query in capability.query_templates.items():
            if normalize_text(key) == dimension_name:
                return query.strip()
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
