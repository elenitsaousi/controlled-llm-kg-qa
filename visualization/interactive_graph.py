from __future__ import annotations

import re
import math
from html import escape
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS
try:
    from pyvis.network import Network
except Exception:  # pragma: no cover
    Network = None  # type: ignore

SURVEY_NS = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"
_PREF_LABEL_RE = re.compile(r"\bsurvey:([A-Za-z0-9_\-./%<>=:]+)")
_ABS_URI_RE = re.compile(r"<(https?://[^>]+)>")
_SURVEY_PREDICATE_RE = re.compile(r"\bsurvey:([A-Za-z0-9_\-]+)\b")
_TECHNICAL_PREDICATES = {
    str(RDF.type),
    str(RDFS.domain),
    str(RDFS.range),
    str(RDFS.label),
}


def _short_term(term) -> str:
    txt = str(term)
    if txt.startswith(SURVEY_NS):
        return txt[len(SURVEY_NS):]
    common = {
        str(RDF.type): "type",
        str(RDFS.domain): "domain",
        str(RDFS.range): "range",
        "http://www.w3.org/2001/XMLSchema#decimal": "decimal",
        "http://www.w3.org/2001/XMLSchema#string": "string",
        "http://www.w3.org/2001/XMLSchema#boolean": "boolean",
    }
    if txt in common:
        return common[txt]
    if "#" in txt:
        return txt.rsplit("#", 1)[-1]
    if "/" in txt:
        return txt.rstrip("/").rsplit("/", 1)[-1]
    return txt


def _node_id(term) -> str:
    return str(term)


def _is_entity(term) -> bool:
    return isinstance(term, URIRef)


def build_graph_html(
    triples: Sequence[Tuple],
    height_px: int = 760,
    heading: str = "Graph View",
    max_nodes: int = 48,
    max_edges: int = 80,
) -> str:
    if Network is None:
        return _build_svg_graph_html(
            triples,
            height_px=height_px,
            heading=heading,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    net = Network(
        height=f"{max(300, int(height_px))}px",
        width="100%",
        directed=True,
        bgcolor="#0f1f28",
        font_color="#edf4f3",
    )
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
                color={
                    "background": "#b9f2f2" if _is_entity(s) else "#54656d",
                    "border": "#7de4df" if _is_entity(s) else "#70838b",
                    "highlight": {
                        "background": "#d8ffff" if _is_entity(s) else "#738891",
                        "border": "#19d6c6",
                    },
                },
                shape="dot" if _is_entity(s) else "box",
            )
            seen_nodes.add(sid)
        if oid not in seen_nodes:
            net.add_node(
                oid,
                label=_short_term(o),
                title=_short_term(o),
                color={
                    "background": "#b9f2f2" if _is_entity(o) else "#54656d",
                    "border": "#7de4df" if _is_entity(o) else "#70838b",
                    "highlight": {
                        "background": "#d8ffff" if _is_entity(o) else "#738891",
                        "border": "#19d6c6",
                    },
                },
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
            "stabilization": {"iterations": 260},
            "barnesHut": {
              "gravitationalConstant": -4200,
              "springLength": 180,
              "springConstant": 0.035,
              "damping": 0.22
            }
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
            "borderWidth": 1.5,
            "size": 22,
            "shadow": {"enabled": true, "color": "rgba(25,214,198,0.18)", "size": 18, "x": 0, "y": 0},
            "font": {"size": 14, "color": "#edf4f3", "face": "Arial"}
          },
          "edges": {
            "smooth": {"type": "curvedCW", "roundness": 0.18},
            "font": {"size": 10, "align": "middle", "color": "#91a4a4"},
            "color": {"color": "#6a8a8f", "opacity": 0.6},
            "width": 1.2
          }
        }
        """
    )
    return net.generate_html(notebook=False)


def _build_svg_graph_html(
    triples: Sequence[Tuple],
    *,
    height_px: int,
    heading: str,
    max_nodes: int = 48,
    max_edges: int = 80,
) -> str:
    width = 1200
    height = max(360, int(height_px))
    margin = 72
    triples = list(triples[:max_edges])

    node_terms: List[object] = []
    seen = set()
    edges: List[Tuple[object, object, object]] = []
    for s, p, o in triples:
        if s not in seen and len(node_terms) < max_nodes:
            node_terms.append(s)
            seen.add(s)
        if o not in seen and len(node_terms) < max_nodes:
            node_terms.append(o)
            seen.add(o)
        if s in seen and o in seen:
            edges.append((s, p, o))
    if not node_terms:
        return f"<h4>{escape(heading)}</h4><p>No graph data available.</p>"

    center_x = width / 2
    center_y = height / 2
    radius = max(90.0, min(width, height) * 0.34)
    positions = {}
    for idx, term in enumerate(node_terms):
        angle = (2 * math.pi * idx) / max(1, len(node_terms))
        positions[term] = (
            center_x + (radius * math.cos(angle)),
            center_y + (radius * math.sin(angle)),
        )
    _apply_force_layout(
        positions,
        [(s, o) for s, _, o in edges if s in positions and o in positions],
        width=width,
        height=height,
        margin=margin,
    )

    edge_markup = []
    edge_labels = []
    for s, p, o in edges:
        if s not in positions or o not in positions:
            continue
        x1, y1 = positions[s]
        x2, y2 = positions[o]
        control_x = (x1 + x2) / 2 + ((y2 - y1) * 0.10)
        control_y = (y1 + y2) / 2 - ((x2 - x1) * 0.10)
        edge_markup.append(
            f"<path d='M {x1:.1f} {y1:.1f} Q {control_x:.1f} {control_y:.1f} {x2:.1f} {y2:.1f}' "
            "fill='none' stroke='#6a8a8f' stroke-opacity='0.72' stroke-width='1.25' marker-end='url(#arrow)' />"
        )
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        edge_labels.append(
            f"<text x='{mx:.1f}' y='{my - 4:.1f}' fill='#91a4a4' font-size='10' "
            f"text-anchor='middle'>{escape(_short_term(p))}</text>"
        )

    node_markup = []
    for term, (x, y) in positions.items():
        is_entity = _is_entity(term)
        fill = "#b9f2f2" if is_entity else "#54656d"
        stroke = "#7de4df" if is_entity else "#70838b"
        label = escape(_short_term(term))
        node_markup.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='21' fill='{fill}' stroke='{stroke}' "
            "stroke-width='2' opacity='0.95' />"
            f"<text x='{x:.1f}' y='{y + 31:.1f}' fill='#edf4f3' font-size='11' "
            f"text-anchor='middle'>{label}</text>"
        )

    return (
        "<style>"
        "html,body{margin:0;background:#101719;color:#edf4f3;font-family:Arial,sans-serif;}"
        "h4{margin:0 0 12px;color:#edf4f3;font-family:Arial,sans-serif;}"
        ".toolbar{display:flex;gap:8px;margin-bottom:12px;}"
        ".toolbar button{background:#172124;color:#edf4f3;border:1px solid #26373a;"
        "border-radius:6px;padding:7px 10px;cursor:pointer;}"
        ".toolbar button:hover{border-color:#19d6c6;}"
        "</style>"
        f"<h4>{escape(heading)}</h4>"
        "<div class='toolbar'>"
        "<button onclick='zoomGraph(1.2)'>Zoom in</button>"
        "<button onclick='zoomGraph(0.83)'>Zoom out</button>"
        "<button onclick='resetGraph()'>Reset</button>"
        "</div>"
        "<div id='graphWrap' style='overflow:hidden; border:1px solid #26373a; border-radius:14px;"
        "background:radial-gradient(circle at 50% 35%, #123342 0%, #0f1f28 58%, #0b151a 100%); cursor:grab'>"
        f"<svg id='graphSvg' viewBox='0 0 {width} {height}' width='100%' height='{height}' "
        "xmlns='http://www.w3.org/2000/svg' role='img'>"
        "<defs><marker id='arrow' markerWidth='8' markerHeight='8' refX='7' refY='3' "
        "orient='auto'><path d='M0,0 L0,6 L8,3 z' fill='#5f7477'/></marker></defs>"
        "<g id='graphViewport'>"
        + "".join(edge_markup)
        + "".join(edge_labels)
        + "".join(node_markup)
        + "</g></svg></div>"
        "<script>"
        "const svg=document.getElementById('graphSvg');"
        "const viewport=document.getElementById('graphViewport');"
        "const wrap=document.getElementById('graphWrap');"
        "let scale=1, tx=0, ty=0, dragging=false, sx=0, sy=0;"
        "function applyGraph(){viewport.setAttribute('transform',`translate(${tx} ${ty}) scale(${scale})`);}"
        "function zoomGraph(f){scale=Math.max(0.25,Math.min(5,scale*f));applyGraph();}"
        "function resetGraph(){scale=1;tx=0;ty=0;applyGraph();}"
        "wrap.addEventListener('wheel',(e)=>{e.preventDefault();zoomGraph(e.deltaY<0?1.1:0.9);},{passive:false});"
        "wrap.addEventListener('mousedown',(e)=>{dragging=true;sx=e.clientX;sy=e.clientY;wrap.style.cursor='grabbing';});"
        "window.addEventListener('mouseup',()=>{dragging=false;wrap.style.cursor='grab';});"
        "window.addEventListener('mousemove',(e)=>{if(!dragging)return;tx+=(e.clientX-sx)/scale;ty+=(e.clientY-sy)/scale;sx=e.clientX;sy=e.clientY;applyGraph();});"
        "</script>"
    )


def _apply_force_layout(
    positions: Dict[object, Tuple[float, float]],
    edges: Sequence[Tuple[object, object]],
    *,
    width: int,
    height: int,
    margin: int,
    iterations: int = 90,
) -> None:
    nodes = list(positions)
    if len(nodes) < 2:
        return
    area = float(width * height)
    k = math.sqrt(area / len(nodes))
    temperature = min(width, height) / 8.0

    for step in range(iterations):
        disp = {node: [0.0, 0.0] for node in nodes}
        for i, v in enumerate(nodes):
            vx, vy = positions[v]
            for u in nodes[i + 1 :]:
                ux, uy = positions[u]
                dx = vx - ux
                dy = vy - uy
                dist = max(0.01, math.hypot(dx, dy))
                force = (k * k) / dist
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                disp[v][0] += fx
                disp[v][1] += fy
                disp[u][0] -= fx
                disp[u][1] -= fy

        for v, u in edges:
            vx, vy = positions[v]
            ux, uy = positions[u]
            dx = vx - ux
            dy = vy - uy
            dist = max(0.01, math.hypot(dx, dy))
            force = (dist * dist) / k
            fx = (dx / dist) * force
            fy = (dy / dist) * force
            disp[v][0] -= fx
            disp[v][1] -= fy
            disp[u][0] += fx
            disp[u][1] += fy

        cooling = temperature * (1.0 - (step / max(1, iterations)))
        for node in nodes:
            x, y = positions[node]
            dx, dy = disp[node]
            dist = max(0.01, math.hypot(dx, dy))
            x += (dx / dist) * min(dist, cooling)
            y += (dy / dist) * min(dist, cooling)
            positions[node] = (
                min(width - margin, max(margin, x)),
                min(height - margin, max(margin, y)),
            )


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


def collect_answer_evidence_triples(
    graph: Graph,
    query: str,
    limit: int = 24,
) -> Tuple[List[Tuple], Dict[str, int]]:
    query_predicates = _query_business_predicates(query)
    out: List[Tuple] = []
    seen = set()

    for predicate in query_predicates:
        domains = [o for _, _, o in graph.triples((predicate, RDFS.domain, None))]
        ranges = [o for _, _, o in graph.triples((predicate, RDFS.range, None))]
        if not domains:
            domains = [URIRef(SURVEY_NS + "Entity")]
        if not ranges:
            ranges = [URIRef(SURVEY_NS + "Value")]
        for domain in domains[:2]:
            for range_ in ranges[:2]:
                triple = (domain, predicate, range_)
                key = tuple(map(str, triple))
                if key in seen:
                    continue
                seen.add(key)
                out.append(triple)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break

    return out, {
        "predicate_count": len(query_predicates),
        "class_count": len({t[0] for t in out} | {t[2] for t in out}),
        "edge_count": len(out),
    }


def _query_business_predicates(query: str) -> List[URIRef]:
    class_terms = set(
        re.findall(r"\ba\s+survey:([A-Za-z0-9_\-]+)\b", query or "")
    )
    seen = set()
    predicates: List[URIRef] = []
    for match in _SURVEY_PREDICATE_RE.finditer(query or ""):
        term = match.group(1)
        if term in class_terms:
            continue
        uri = URIRef(SURVEY_NS + term)
        if uri in seen:
            continue
        seen.add(uri)
        predicates.append(uri)
    return predicates
