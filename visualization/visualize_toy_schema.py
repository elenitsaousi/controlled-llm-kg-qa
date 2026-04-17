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
    for label in schema.get("labels", {}).keys():
        G.add_node(label, name=label)

    for rel in schema.get("relationships", []):
        rel_type = rel.get("type")
        for src in rel.get("from", []):
            for dst in rel.get("to", []):
                G.add_edge(src, dst, weight=1, label=rel_type)
    return G


def main() -> None:
    schema_path = os.path.join(base_dir, "data", "toy_kg", "schema.json")
    schema = _load_schema(schema_path)
    graph = _build_schema_graph(schema)

    visualizer = GraphVisualizer()
    visualizer.visualize_existing_graph_interactive(
        graph,
        node_scale=10,
        figsize=(18, 12),
        weight_threshold=0,
        label_top_n=50,
        iterations=500,
        show_names=True,
        layout_scale=0.75,
        label_font_size=10,
    )


if __name__ == "__main__":
    main()
