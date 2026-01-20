import math

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

try:
    from fa2_modified import ForceAtlas2
except ImportError:
    ForceAtlas2 = None


class GraphVisualizer:
    def _normalize_node_sizes(self, values, new_min=5, new_max=100):
        if not values or len(values) == 0:
            return []

        old_min = min(values)
        old_max = max(values)

        if old_min == old_max:
            return [new_min for _ in values]

        return [
            new_min + (val - old_min) * (new_max - new_min) / (old_max - old_min)
            for val in values
        ]

    def _filter_and_prepare_graph(self, G, node_scale, weight_threshold):
        filtered_edges = [
            (u, v) for u, v, w in G.edges(data="weight") if w > weight_threshold
        ]
        H = G.edge_subgraph(filtered_edges).copy()

        if len(H.nodes()) == 0:
            print("Warning: Graph has no nodes after filtering. Cannot visualize.")
            return None, None, None

        node_weights = {node: 0 for node in H.nodes()}
        for u, v, w in H.edges(data="weight"):
            node_weights[u] += w
            node_weights[v] += w

        raw_node_sizes = [node_weights[n] * node_scale for n in H.nodes()]
        node_sizes = self._normalize_node_sizes(values=raw_node_sizes)

        if not node_sizes:
            print("Warning: No valid node sizes computed. Cannot visualize.")
            return None, None, None

        return H, node_weights, node_sizes

    def _calculate_layout(self, H, cluster_colors, iterations):
        if ForceAtlas2 is None:
            return nx.spring_layout(H, seed=42)

        forceatlas2 = ForceAtlas2(
            outboundAttractionDistribution=False,
            linLogMode=False,
            adjustSizes=True,
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            strongGravityMode=False,
            verbose=True,
        )

        initial_pos = None

        if cluster_colors is not None:
            print("Adapting layout for cluster display...")

            cluster_groups = {}
            for node in H.nodes():
                cluster_id = cluster_colors.get(node, -1)
                cluster_groups.setdefault(cluster_id, []).append(node)

            num_clusters = len([c for c in cluster_groups.keys() if c >= 0])

            if num_clusters > 0:
                initial_pos = {}
                angle_step = 2 * math.pi / num_clusters if num_clusters > 1 else 0
                cluster_idx = 0

                for cluster_id, nodes in cluster_groups.items():
                    if cluster_id >= 0 and len(nodes) > 0:
                        angle = cluster_idx * angle_step
                        center_x = 100 * math.cos(angle)
                        center_y = 100 * math.sin(angle)
                        radius = min(30, 5 * math.sqrt(len(nodes)))

                        node_angle_step = (
                            2 * math.pi / len(nodes) if len(nodes) > 1 else 0
                        )
                        for i, node in enumerate(nodes):
                            node_angle = i * node_angle_step
                            initial_pos[node] = (
                                center_x + radius * math.cos(node_angle),
                                center_y + radius * math.sin(node_angle),
                            )
                        cluster_idx += 1

                if -1 in cluster_groups:
                    for node in cluster_groups[-1]:
                        initial_pos[node] = (0, 0)

            forceatlas2.edgeWeightInfluence = 1.2
            forceatlas2.scalingRatio = 120.0
            forceatlas2.gravity = 0.15
        else:
            forceatlas2.edgeWeightInfluence = 1
            forceatlas2.scalingRatio = 100.0
            forceatlas2.gravity = 0.1

        pos = forceatlas2.forceatlas2_networkx_layout(
            H, pos=initial_pos, iterations=iterations
        )
        return pos

    def _draw_graph(self, H, pos, node_sizes, cluster_colors, figsize):
        fig, ax = plt.subplots(figsize=figsize)

        if cluster_colors is not None:
            node_colors = [cluster_colors.get(node, 0) for node in H.nodes()]
            cmap = cm.get_cmap("tab20")
            node_colors_mapped = [cmap(c % 20) for c in node_colors]
            title = "Weighted Network Graph (Clustered)"

            unique_cluster_ids = sorted(list(set(cluster_colors.values())))
            legend_handles = []
            legend_labels = []

            for cluster_id in unique_cluster_ids:
                if cluster_id >= 0:
                    color = cmap(cluster_id % 20)
                    legend_handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=7,
                            label=f"Cluster {cluster_id}",
                        )
                    )
                    legend_labels.append(f"Cluster {cluster_id}")

            intra_cluster_edges = []
            inter_cluster_edges = []
            for u, v in H.edges():
                u_c = cluster_colors.get(u, -1)
                v_c = cluster_colors.get(v, -1)
                if u_c == v_c and u_c >= 0:
                    intra_cluster_edges.append((u, v))
                else:
                    inter_cluster_edges.append((u, v))

            if inter_cluster_edges:
                nx.draw_networkx_edges(
                    H,
                    pos,
                    edgelist=inter_cluster_edges,
                    edge_color="lightgray",
                    width=0.15,
                    alpha=0.3,
                    ax=ax,
                    style="dashed",
                )

            if intra_cluster_edges:
                nx.draw_networkx_edges(
                    H,
                    pos,
                    edgelist=intra_cluster_edges,
                    edge_color="gray",
                    width=0.3,
                    alpha=0.6,
                    ax=ax,
                )

            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    title="Clusters",
                    loc="upper left",
                    bbox_to_anchor=(0.98, 1.00),
                    frameon=False,
                    fontsize=7,
                )

        else:
            node_colors_mapped = "skyblue"
            title = "Weighted Network Graph"

            nx.draw_networkx_edges(
                H, pos, edge_color="gray", width=0.2, alpha=0.5, ax=ax
            )

        nx.draw_networkx_nodes(
            H,
            pos,
            node_size=node_sizes,
            node_color=node_colors_mapped,
            alpha=0.7,
            ax=ax,
        )

        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()

        return fig, ax

    def _setup_interactivity(
        self, fig, ax, H, pos, node_weights, label_top_n, show_names
    ):
        def update_labels():
            for artist in ax.texts:
                artist.remove()

            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            visible_nodes = [
                node
                for node, (x, y) in pos.items()
                if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]
            ]

            top_nodes = sorted(
                [(n, node_weights[n]) for n in visible_nodes],
                key=lambda x: x[1],
                reverse=True,
            )[:label_top_n]

            labels = {}
            if show_names:
                for node, _ in top_nodes:
                    node_data = H.nodes[node]
                    label = node_data.get("name", node)
                    if label == node and node.startswith("author_"):
                        label = node_data.get("name", node.replace("author_", ""))

                    labels[node] = label

            nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, ax=ax)

        def zoom(event):
            base_scale = 1.1
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            if xdata is None or ydata is None:
                return

            scale_factor = (
                1 / base_scale
                if event.button == "up"
                else base_scale if event.button == "down" else 1
            )

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

            update_labels()
            fig.canvas.draw_idle()

        def pan(event):
            if (
                event.button == 1
                and event.inaxes == ax
                and pan.prev_x is not None
                and pan.prev_y is not None
            ):
                dx = -event.xdata + pan.prev_x
                dy = -event.ydata + pan.prev_y

                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()

                ax.set_xlim(cur_xlim + dx)
                ax.set_ylim(cur_ylim + dy)

                update_labels()
                fig.canvas.draw_idle()

            pan.prev_x, pan.prev_y = event.xdata, event.ydata

        pan.prev_x, pan.prev_y = None, None

        fig.canvas.mpl_connect("scroll_event", zoom)
        fig.canvas.mpl_connect("motion_notify_event", pan)

        update_labels()

    def _draw_centrality_graph(
        self, H, pos, node_sizes, centrality_measures, measure_name, figsize
    ):
        fig, ax = plt.subplots(figsize=figsize)

        node_values = [centrality_measures.get(node, 0) for node in H.nodes()]

        nx.draw_networkx_edges(H, pos, edge_color="gray", width=0.2, alpha=0.5, ax=ax)

        nodes = nx.draw_networkx_nodes(
            H,
            pos,
            node_size=node_sizes,
            node_color=node_values,
            cmap=plt.cm.plasma,
            alpha=0.7,
            ax=ax,
        )

        nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
        plt.colorbar(nodes, ax=ax, label=measure_name)

        ax.set_title(f"Network Graph colored by {measure_name}", fontsize=14)
        ax.axis("off")
        plt.tight_layout()

        return fig, ax

    def visualize_existing_graph_interactive(
        self,
        G,
        node_scale=5,
        figsize=(24, 16),
        weight_threshold=0,
        label_top_n=50,
        iterations=1000,
        show_names=False,
        cluster_colors=None,
        measure_name=None,
        centrality_measures=None,
    ):
        H, node_weights, node_sizes = self._filter_and_prepare_graph(
            G, node_scale, weight_threshold
        )

        if not H or len(H.nodes()) == 0:
            return

        pos = self._calculate_layout(H, cluster_colors, iterations)
        if centrality_measures is not None and measure_name is not None:
            fig, ax = self._draw_centrality_graph(
                H, pos, node_sizes, centrality_measures, measure_name, figsize
            )
        else:
            fig, ax = self._draw_graph(H, pos, node_sizes, cluster_colors, figsize)

        self._setup_interactivity(
            fig, ax, H, pos, node_weights, label_top_n, show_names
        )

        plt.show(block=True)
        return H
