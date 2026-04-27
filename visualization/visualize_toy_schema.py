import json
import os
import sys

import networkx as nx

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from visualization.graph_visualizer import GraphVisualizer


def _load_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_schema_graph(schema: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for label, metadata in schema.get("labels", {}).items():
        domains = metadata.get("domain", [])
        if len(domains) > 1:
            domain = "hybrid"
        elif domains:
            domain = domains[0]
        else:
            domain = "other"
        G.add_node(label, name=label, domain=domain)

    for rel in schema.get("relationships", []):
        rel_type = rel.get("type")
        edge_label = "HAS\nCONSTRAINT" if rel_type == "HAS_CONSTRAINT" else rel_type
        for src in rel.get("from", []):
            for dst in rel.get("to", []):
                G.add_edge(src, dst, weight=1, label=edge_label)
    return G


def main() -> None:
    schema_path = os.path.join(base_dir, "data", "toy_kg", "schema.json")
    schema = _load_schema(schema_path)
    graph = _build_schema_graph(schema)

    visualizer = GraphVisualizer()
    visualizer.visualize_existing_graph_interactive(
        graph,
        node_scale=80,
        figsize=(20, 13),
        weight_threshold=0,
        label_top_n=50,
        iterations=500,
        show_names=True,
        layout_scale=1.0,
        label_font_size=16,
        node_size_range=(950, 1850),
        node_color_attr="domain",
        node_palette={
            "manufacturing": "#86efac",
            "supply_chain": "#93c5fd",
            "hybrid": "#fbbf24",
            "other": "#d1d5db",
            "default": "#d1d5db",
        },
        show_edge_labels=True,
        edge_label_font_size=11,
        manual_node_positions={
            "Supplier": (-4.0, 1.0),
            "Material": (-4.0, -1.0),
            "Yield": (-2.2, 2.4),
            "Inventory": (-2.0, -2.0),
            "Product": (0.0, -0.25),
            "ProcessStep": (0.0, -2.3),
            "Fab": (2.2, -1.25),
            "CapacityConstraint": (5.25, -1.25),
            "Order": (0.6, 1.45),
            "Lot": (2.0, 1.0),
            "Tool": (1.1, 3.1),
            "Shipment": (3.2, 2.1),
            "Status": (4.6, 3.0),
            "Defect": (4.4, 0.45),
        },
        output_path=os.path.join(base_dir, "analysis_outputs", "toy_schema_preview.png"),
    )


if __name__ == "__main__":
    main()
