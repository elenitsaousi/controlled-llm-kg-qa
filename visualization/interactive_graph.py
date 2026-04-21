from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdflib import Graph, URIRef
try:
    from pyvis.network import Network
except Exception:  # pragma: no cover
    Network = None  # type: ignore

SURVEY_NS = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"
_PREF_LABEL_RE = re.compile(r"\bsurvey:([A-Za-z0-9_\-./%<>=:]+)")
_ABS_URI_RE = re.compile(r"<(https?://[^>]+)>")


def _short_term(term) -> str:
    txt = str(term)
    if txt.startswith(SURVEY_NS):
        return "survey:" + txt[len(SURVEY_NS):]
    return txt


def _node_id(term) -> str:
    return str(term)


def _is_entity(term) -> bool:
    return isinstance(term, URIRef)


def build_graph_html(
    triples: Sequence[Tuple],
    height_px: int = 760,
    heading: str = "Graph View",
) -> str:
    if Network is None:
        rows = "".join(
            f"<tr><td>{_short_term(s)}</td><td>{_short_term(p)}</td><td>{_short_term(o)}</td></tr>"
            for s, p, o in triples[:500]
        )
        return (
            f"<h4>{heading}</h4>"
            "<p>pyvis is not installed; showing triple table preview.</p>"
            "<table border='1' cellpadding='4' cellspacing='0'>"
            "<tr><th>Subject</th><th>Predicate</th><th>Object</th></tr>"
            f"{rows}</table>"
        )

    net = Network(height=f"{max(300, int(height_px))}px", width="100%", directed=True)
    net.barnes_hut()
    net.heading = heading
    seen_nodes = set()
    seen_edges = set()

    for s, p, o in triples:
        sid = _node_id(s)
        oid = _node_id(o)
        if sid not in seen_nodes:
            net.add_node(
                sid,
                label=_short_term(s),
                title=_short_term(s),
                color="#1f77b4" if _is_entity(s) else "#7f7f7f",
                shape="dot" if _is_entity(s) else "box",
            )
            seen_nodes.add(sid)
        if oid not in seen_nodes:
            net.add_node(
                oid,
                label=_short_term(o),
                title=_short_term(o),
                color="#ff7f0e" if _is_entity(o) else "#7f7f7f",
                shape="dot" if _is_entity(o) else "box",
            )
            seen_nodes.add(oid)

        edge_key = (sid, _short_term(p), oid)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        net.add_edge(sid, oid, label=_short_term(p), title=_short_term(p), arrows="to")

    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 300}
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true,
            "zoomView": true,
            "dragView": true
          },
          "nodes": {
            "borderWidth": 1,
            "size": 16,
            "font": {"size": 14}
          },
          "edges": {
            "smooth": {"type": "dynamic"},
            "font": {"size": 10, "align": "middle"},
            "color": {"opacity": 0.7}
          }
        }
        """
    )
    return net.generate_html(notebook=False)


def collect_full_graph_triples(graph: Graph, limit: int = 3000) -> Tuple[List[Tuple], int]:
    total = 0
    out: List[Tuple] = []
    use_limit = int(limit) > 0
    for triple in graph:
        total += 1
        if use_limit and len(out) >= int(limit):
            continue
        out.append(triple)
    return out, total


def _query_seed_uris(query: str) -> List[URIRef]:
    seeds: List[URIRef] = []
    for m in _PREF_LABEL_RE.finditer(query or ""):
        local = m.group(1).strip()
        if not local:
            continue
        seeds.append(URIRef(SURVEY_NS + local))
    for m in _ABS_URI_RE.finditer(query or ""):
        seeds.append(URIRef(m.group(1).strip()))
    return seeds


def _rows_seed_uris(rows: Iterable[Dict[str, str]]) -> List[URIRef]:
    seeds: List[URIRef] = []
    for row in rows:
        for value in row.values():
            txt = str(value).strip()
            if txt.startswith("http://") or txt.startswith("https://"):
                seeds.append(URIRef(txt))
    return seeds


def collect_query_subgraph_triples(
    graph: Graph,
    query: str,
    result_rows: Optional[Iterable[Dict[str, str]]] = None,
    hops: int = 1,
    limit: int = 1200,
) -> Tuple[List[Tuple], Dict[str, int]]:
    max_hops = max(1, int(hops))
    max_edges = max(1, int(limit))
    seeds = set(_query_seed_uris(query))
    if result_rows is not None:
        seeds.update(_rows_seed_uris(result_rows))
    if not seeds:
        return [], {"seed_count": 0, "edge_count": 0}

    out: List[Tuple] = []
    seen_edges = set()
    frontier = set(seeds)
    visited = set(seeds)

    for _ in range(max_hops):
        if not frontier or len(out) >= max_edges:
            break
        next_frontier = set()
        for node in list(frontier):
            for s, p, o in graph.triples((node, None, None)):
                edge_key = (_node_id(s), _short_term(p), _node_id(o))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    out.append((s, p, o))
                    if len(out) >= max_edges:
                        break
                if isinstance(o, URIRef) and o not in visited:
                    next_frontier.add(o)
                    visited.add(o)
            if len(out) >= max_edges:
                break
            for s, p, o in graph.triples((None, None, node)):
                edge_key = (_node_id(s), _short_term(p), _node_id(o))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    out.append((s, p, o))
                    if len(out) >= max_edges:
                        break
                if isinstance(s, URIRef) and s not in visited:
                    next_frontier.add(s)
                    visited.add(s)
            if len(out) >= max_edges:
                break
        frontier = next_frontier

    return out, {"seed_count": len(seeds), "edge_count": len(out)}
