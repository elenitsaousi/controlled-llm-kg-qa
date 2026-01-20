import json
import os
import sys

import networkx as nx

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from visualization.graph_visualizer import GraphVisualizer


def _load_instances(instances_path: str) -> dict:
    with open(instances_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_instance_graph(instances: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in instances.get("nodes", []):
        label = node.get("label", "Node")
        name = node.get("name", node.get("id"))
        display_name = f"{label}:{name}"
        G.add_node(node["id"], name=display_name, label=label)

    for edge in instances.get("edges", []):
        G.add_edge(edge["from"], edge["to"], weight=1, label=edge.get("type"))
    return G


def main() -> None:
    instances_path = os.path.join(
        base_dir, "data", "toy_kg", "instances.json"
    )
    instances = _load_instances(instances_path)
    graph = _build_instance_graph(instances)

    visualizer = GraphVisualizer()
    visualizer.visualize_existing_graph_interactive(
        graph,
        node_scale=8,
        figsize=(20, 14),
        weight_threshold=0,
        label_top_n=80,
        iterations=600,
        show_names=True,
    )


if __name__ == "__main__":
    main()
