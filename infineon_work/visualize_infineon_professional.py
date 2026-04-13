import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, RDF, RDFS, URIRef, Literal

BASES = {
    "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/",
    "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey#",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_TTL = PROJECT_ROOT / "data" / "infineon" / "graph.ttl"
COUNTS_JSON = PROJECT_ROOT / "data" / "infineon" / "class_graph_counts.json"


def _local_name(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#", 1)[1]
    return s.rsplit("/", 1)[-1]


def _in_base(uri: URIRef) -> bool:
    uri_str = str(uri)
    return any(uri_str.startswith(base) for base in BASES)


def _scale(values, new_min, new_max):
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_min == v_max:
        return [new_min for _ in values]
    return [
        new_min + (v - v_min) * (new_max - new_min) / (v_max - v_min)
        for v in values
    ]


def build_counts() -> dict:
    g = Graph()
    g.parse(str(GRAPH_TTL), format="turtle")

    type_map = defaultdict(set)
    class_counts = defaultdict(int)

    base_uris = set()
    base_subjects = set()
    base_objects = set()

    for s, p, o in g:
        if isinstance(s, URIRef) and _in_base(s):
            base_subjects.add(s)
            base_uris.add(s)
        if isinstance(o, URIRef) and _in_base(o):
            base_objects.add(o)
            base_uris.add(o)

    for s, _, o in g.triples((None, RDF.type, None)):
        if _in_base(o):
            cls = _local_name(o)
            type_map[s].add(cls)
            class_counts[cls] += 1
        elif o == RDFS.Class and _in_base(s):
            cls = _local_name(s)
            type_map[s].add(cls)

    edge_counts = defaultdict(int)
    predicate_counts = defaultdict(int)

    for s, p, o in g:
        if not _in_base(p):
            continue

        pred = _local_name(p)
        predicate_counts[pred] += 1

        if isinstance(o, Literal):
            continue
        if not (isinstance(s, URIRef) and isinstance(o, URIRef)):
            continue

        src_types = set(type_map.get(s, set()))
        dst_types = set(type_map.get(o, set()))

        if not src_types and _in_base(s):
            src_types = {_local_name(s)}
        if not dst_types and _in_base(o):
            dst_types = {_local_name(o)}

        if not src_types or not dst_types:
            continue

        for st in src_types:
            for dt in dst_types:
                edge_counts[(st, dt)] += 1

    classes = sorted(
        set(class_counts.keys())
        | {s for s, _ in edge_counts.keys()}
        | {t for _, t in edge_counts.keys()}
    )

    data = {
        "base": sorted(BASES),
        "triple_count": len(g),
        "entity_count": len(base_uris),
        "base_subjects": len(base_subjects),
        "base_objects": len(base_objects),
        "classes": [
            {"id": c, "instance_count": int(class_counts.get(c, 0))}
            for c in classes
        ],
        "edges": [
            {"source": s, "target": t, "count": int(cnt)}
            for (s, t), cnt in edge_counts.items()
        ],
        "predicate_counts": [
            {"predicate": p, "count": int(cnt)}
            for p, cnt in predicate_counts.items()
        ],
    }

    COUNTS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def load_counts(rebuild: bool) -> dict:
    if rebuild or not COUNTS_JSON.exists():
        return build_counts()
    with COUNTS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(
    data: dict,
    min_edge_count: int,
    top_edges: int | None,
    min_node_count: int,
    largest_component: bool,
) -> nx.Graph:
    edges = [
        (e["source"], e["target"], int(e["count"]))
        for e in data.get("edges", [])
    ]

    if min_edge_count > 0:
        edges = [e for e in edges if e[2] >= min_edge_count]

    if top_edges:
        edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_edges]

    node_counts = {c["id"]: int(c.get("instance_count", 0)) for c in data["classes"]}

    G = nx.Graph()
    for node, cnt in node_counts.items():
        if cnt >= min_node_count:
            G.add_node(node, count=cnt)

    for s, t, cnt in edges:
        if s in G and t in G:
            G.add_edge(s, t, weight=cnt)

    if largest_component and len(G) > 0:
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    return G


def draw_graph(
    G: nx.Graph,
    output_path: Path,
    label_top_n: int,
    seed: int,
):
    if len(G) == 0:
        raise SystemExit("Graph is empty after filtering.")

    counts = [G.nodes[n].get("count", 1) for n in G.nodes()]
    node_sizes = _scale([math.log1p(c) for c in counts], 200, 1600)

    edge_weights = [G.edges[e].get("weight", 1) for e in G.edges()]
    edge_widths = _scale([math.log1p(w) for w in edge_weights], 0.2, 2.8)

    pos = nx.spring_layout(G, seed=seed, k=1 / math.sqrt(len(G)))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Color by component
    components = list(nx.connected_components(G))
    comp_id = {}
    for i, comp in enumerate(components):
        for n in comp:
            comp_id[n] = i
    cmap = plt.get_cmap("tab20")
    node_colors = [cmap(comp_id[n] % 20) for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.25, edge_color="#444")
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=0.3,
        edgecolors="#333",
        alpha=0.9,
    )

    # Label only top-N by instance count
    top_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n].get("count", 0), reverse=True)
    labels = {n: n for n in top_nodes[:label_top_n]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#111")

    ax.set_title("Infineon Class Graph (Observed Relations)")
    ax.axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Professional visualization of Infineon class graph."
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild counts from graph.ttl")
    parser.add_argument("--min-edge-count", type=int, default=1)
    parser.add_argument("--top-edges", type=int, default=0)
    parser.add_argument("--min-node-count", type=int, default=1)
    parser.add_argument("--largest-component", action="store_true")
    parser.add_argument("--label-top-n", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "infineon_work" / "figures" / "infineon_class_graph.svg"),
    )

    args = parser.parse_args()

    data = load_counts(rebuild=args.rebuild)
    G = build_graph(
        data,
        min_edge_count=args.min_edge_count,
        top_edges=args.top_edges if args.top_edges > 0 else None,
        min_node_count=args.min_node_count,
        largest_component=args.largest_component,
    )
    draw_graph(G, Path(args.output), label_top_n=args.label_top_n, seed=args.seed)

    components = list(nx.connected_components(G)) if len(G) > 0 else []
    summary = {
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "components": len(components),
        "largest_component": max((len(c) for c in components), default=0),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
