import json
import os
import sys

import networkx as nx

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from visualization.graph_visualizer import GraphVisualizer


def _load_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_schema_graph(schema: dict, include_properties: bool = False) -> nx.DiGraph:
    G = nx.DiGraph()

    classes = schema.get("classes")
    if not classes:
        classes = list(schema.get("labels", {}).keys())

    for label in classes:
        G.add_node(label, name=label, kind="class")

    if include_properties:
        for prop in schema.get("properties", []):
            prop_node = f"prop:{prop}"
            G.add_node(prop_node, name=prop, kind="property")

    for rel in schema.get("relationships", []):
        rel_type = rel.get("type")
        from_list = rel.get("from", [])
        to_list = rel.get("to", [])

        if to_list:
            for src in from_list:
                for dst in to_list:
                    G.add_edge(src, dst, weight=1, label=rel_type)
        elif include_properties and from_list and rel_type:
            prop_node = f"prop:{rel_type}"
            if prop_node not in G:
                G.add_node(prop_node, name=rel_type, kind="property")
            for src in from_list:
                G.add_edge(src, prop_node, weight=1, label=rel_type)

    return G


def main() -> None:
    schema_path = os.path.join(BASE_DIR, "data", "infineon", "schema.json")
    include_properties = "--with-properties" in sys.argv
    use_cluster = "--cluster" in sys.argv

    schema = _load_schema(schema_path)
    graph = _build_schema_graph(schema, include_properties=include_properties)

    cluster_colors = None
    if include_properties and use_cluster:
        cluster_colors = {
            node: (0 if graph.nodes[node].get("kind") == "class" else 1)
            for node in graph.nodes()
        }

    visualizer = GraphVisualizer()
    visualizer.visualize_existing_graph_interactive(
        graph,
        node_scale=10,
        figsize=(18, 12),
        weight_threshold=0,
        label_top_n=50,
        iterations=500,
        show_names=True,
        cluster_colors=cluster_colors,
    )


if __name__ == "__main__":
    main()
