from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from pathlib import Path
import csv
import re


TTL_PATH = Path("/Users/elenetsaouse/Documents/controlled-llm-kg-qa/data/infineon/graph.ttl")

OUT_DIR = Path("/Users/elenetsaouse/Documents/controlled-llm-kg-qa/data/infineon/gephi_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGES_CSV = OUT_DIR / "gephi_edges.csv"
NODES_CSV = OUT_DIR / "gephi_nodes.csv"


# Keep only useful semantic relations for visualization.
# Do NOT include rdfs:label, rdf:type, numeric/literal properties etc.
KEEP_PREDICATES = {
    "quarter",
    "forTimePeriod",
    "inRegion",
    "fromCompany",
    "forCompany",
    "hasVehicleType",
    "appliesToVehicleType",
    "analyzesVehicleType",
    "hasSAELevel",
    "hasTechnologyCategory",
    "forTechnologyCategory",
    "analyzesTechnologyCategory",
    "hasFutureDemand",
    "hasInventoryTrend",
    "hasInventoryResponse",
    "hasOrderCancellation",
    "hasTargetIndicator",
    "observedInMarket",
    "hasMarketSegment",
    "hasResponseType",
    "hasComponentTypeSplit",
    "hasAggregatedResult",
    "hasDetail",
}

# Optional: keep only subjects/objects that are related to these keywords.
# Set to [] if you want all kept predicates.
FOCUS_KEYWORDS = [
    "FutureDemand",
    "Demand",
    "Quarter",
    "Region",
    "OEM",
    "Tier1",
    "Semiconductor",
]


def local_name(uri) -> str:
    """
    Convert full URI to readable local name.
    """
    text = str(uri)

    if "#" in text:
        text = text.split("#")[-1]
    else:
        text = text.rstrip("/").split("/")[-1]

    # Decode some common URL encodings for labels.
    text = (
        text.replace("%28", "(")
            .replace("%29", ")")
            .replace("%20", "_")
            .replace("%2F", "_")
    )

    return text


def safe_id(text: str) -> str:
    """
    Make IDs safer for Gephi CSV.
    Keeps labels readable but avoids weird separators.
    """
    text = str(text)
    text = text.strip()
    text = text.replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = re.sub(r"[\n\r\t]+", "_", text)
    return text


def matches_focus(source_label: str, target_label: str, predicate_label: str) -> bool:
    """
    Keep edge if focus keywords are disabled, or if any keyword appears in source/target/predicate.
    """
    if not FOCUS_KEYWORDS:
        return True

    combined = f"{source_label} {target_label} {predicate_label}".lower()
    return any(keyword.lower() in combined for keyword in FOCUS_KEYWORDS)


def main():
    print(f"Loading TTL from: {TTL_PATH}")

    if not TTL_PATH.exists():
        raise FileNotFoundError(f"TTL file not found: {TTL_PATH}")

    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")

    print(f"Loaded triples: {len(g):,}")

    edges = []
    nodes = {}

    skipped_literal_objects = 0
    skipped_predicates = 0
    skipped_focus = 0

    for s, p, o in g:
        # Gephi edges need resource -> resource.
        # Skip literal values like numbers, labels, booleans, percentages.
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            if isinstance(o, Literal):
                skipped_literal_objects += 1
            continue

        pred = local_name(p)

        # Skip generic ontology/schema relations and unwanted predicates.
        if pred not in KEEP_PREDICATES:
            skipped_predicates += 1
            continue

        source_label = local_name(s)
        target_label = local_name(o)
        predicate_label = pred

        if not matches_focus(source_label, target_label, predicate_label):
            skipped_focus += 1
            continue

        source_id = safe_id(source_label)
        target_id = safe_id(target_label)

        edges.append((source_id, target_id, predicate_label))

        nodes[source_id] = source_label
        nodes[target_id] = target_label

    # Remove duplicate edges
    edges = sorted(set(edges))

    # Write edges CSV
    with open(EDGES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target", "Type"])
        writer.writerows(edges)

    # Write nodes CSV
    with open(NODES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        for node_id, label in sorted(nodes.items()):
            writer.writerow([node_id, label])

    print("\nExport completed.")
    print(f"Nodes: {len(nodes):,}")
    print(f"Edges: {len(edges):,}")
    print(f"Skipped literal objects: {skipped_literal_objects:,}")
    print(f"Skipped predicates outside whitelist: {skipped_predicates:,}")
    print(f"Skipped by focus filter: {skipped_focus:,}")

    print(f"\nCreated:")
    print(f"- {NODES_CSV}")
    print(f"- {EDGES_CSV}")

    print("\nImport in Gephi:")
    print("1. File → Import Spreadsheet → gephi_nodes.csv → Nodes table")
    print("2. File → Import Spreadsheet → gephi_edges.csv → Edges table")
    print("3. Graph type: Directed")
    print("4. Layout: ForceAtlas2")


if __name__ == "__main__":
    main()