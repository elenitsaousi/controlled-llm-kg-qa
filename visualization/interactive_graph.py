from __future__ import annotations

import re
import math
import json
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
    if isinstance(term, URIRef):
        return True
    if isinstance(term, str):
        value = term.strip()
        if not value:
            return False
        if value.startswith(("http://", "https://", "survey:")):
            return True
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", value):
            return False
        if len(value) <= 80 and not re.search(r"\s", value):
            return True
    return False


def _node_kind(term, info: Optional[Dict[str, object]] = None) -> str:
    if not _is_entity(term):
        return "literal"
    label = _short_term(term)
    uri = str(term)
    types = {str(t).lower() for t in ((info or {}).get("types") or [])}
    if uri.startswith("http://www.w3.org/2001/XMLSchema#") or label in {"decimal", "string", "boolean"}:
        return "datatype"
    if any("property" in t for t in types):
        return "property"
    if any("class" in t for t in types) or label[:1].isupper() or isinstance(term, str):
        return "class"
    return "entity"


def _node_visual_label(term) -> str:
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", _short_term(term)).replace("_", " ")
    words = label.split()
    if len(label) <= 20 or len(words) <= 1:
        return label
    lines: List[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 15 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
        if len(lines) >= 2:
            break
    if line and len(lines) < 3:
        lines.append(line)
    return "\n".join(lines[:3])


def _node_visual_style(term, info: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    kind = _node_kind(term, info)
    if kind == "literal":
        return {
            "shape": "box",
            "size": 16,
            "color": {
                "background": "#d3d7d9",
                "border": "#8c969c",
                "highlight": {"background": "#ff4b4b", "border": "#d50000"},
            },
        }
    if kind == "datatype":
        return {
            "shape": "dot",
            "size": 18,
            "color": {
                "background": "#e2e4e5",
                "border": "#9ca4aa",
                "highlight": {"background": "#ff4b4b", "border": "#d50000"},
            },
        }
    return {
        "shape": "dot",
        "size": 25,
        "color": {
            "background": "#b8d8ff",
            "border": "#4e83c8",
            "highlight": {"background": "#ff4b4b", "border": "#d50000"},
        },
    }


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
        bgcolor="#ffffff",
        font_color="#17262b",
    )
    net.force_atlas_2based()
    net.heading = heading
    seen_nodes = set()
    seen_edges = set()
    node_info: Dict[str, Dict[str, object]] = {}

    def ensure_info(term) -> Dict[str, object]:
        nid = _node_id(term)
        info = node_info.setdefault(
            nid,
            {
                "id": nid,
                "label": _short_term(term),
                "uri": str(term),
                "kind": _node_kind(term),
                "types": [],
                "labels": [],
                "comments": [],
                "outgoing": [],
                "incoming": [],
            },
        )
        return info

    for s, p, o in triples:
        sid = _node_id(s)
        oid = _node_id(o)
        sinfo = ensure_info(s)
        oinfo = ensure_info(o)
        pred_label = _short_term(p)
        if str(p) == str(RDFS.label):
            sinfo.setdefault("labels", []).append(str(o))
        elif str(p) == str(RDFS.comment):
            sinfo.setdefault("comments", []).append(str(o))
        elif str(p) == str(RDF.type):
            sinfo.setdefault("types", []).append(_short_term(o))
        else:
            sinfo.setdefault("outgoing", []).append({"predicate": pred_label, "target": _short_term(o)})
            oinfo.setdefault("incoming", []).append({"predicate": pred_label, "source": _short_term(s)})
        if sid not in seen_nodes:
            style = _node_visual_style(s, sinfo)
            net.add_node(
                sid,
                label=_node_visual_label(s),
                title=_short_term(s),
                color=style["color"],
                shape=style["shape"],
                size=style["size"],
            )
            seen_nodes.add(sid)
        if oid not in seen_nodes:
            style = _node_visual_style(o, oinfo)
            net.add_node(
                oid,
                label=_node_visual_label(o),
                title=_short_term(o),
                color=style["color"],
                shape=style["shape"],
                size=style["size"],
            )
            seen_nodes.add(oid)

        edge_key = (sid, _short_term(p), oid)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        pred_short = _short_term(p)
        net.add_edge(
            sid,
            oid,
            label=pred_short,
            title=pred_short,
            arrows="to",
            dashes=pred_short in {"type", "subClassOf", "domain", "range"},
        )

    net.set_options(
        """
        {
          "layout": {
            "improvedLayout": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {"enabled": true, "iterations": 420, "fit": true},
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -95,
              "centralGravity": 0.006,
              "springLength": 285,
              "springConstant": 0.028,
              "damping": 0.72,
              "avoidOverlap": 1.0
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "dragNodes": true,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true,
            "zoomView": true,
            "dragView": true
          },
          "nodes": {
            "borderWidth": 1.5,
            "size": 20,
            "shadow": {"enabled": true, "color": "rgba(25,214,198,0.18)", "size": 18, "x": 0, "y": 0},
            "font": {"size": 13, "color": "#17262b", "face": "Arial", "strokeWidth": 4, "strokeColor": "#ffffff"}
          },
          "edges": {
            "smooth": {"enabled": true, "type": "dynamic", "roundness": 0.16},
            "font": {"size": 10, "align": "middle", "color": "#2f5f91", "background": "rgba(230,241,255,0.92)", "strokeWidth": 2, "strokeColor": "#e6f1ff"},
            "color": {"color": "#7d8995", "highlight": "#ff4b4b", "hover": "#0067b1", "opacity": 0.74},
            "selectionWidth": 2,
            "width": 1.1
          }
        }
        """
    )
    html = net.generate_html(notebook=False)
    graph_stats = {
        "nodes": len(seen_nodes),
        "edges": len(seen_edges),
        "classes": sum(1 for info in node_info.values() if str(info.get("kind")) == "class"),
        "literals": sum(1 for info in node_info.values() if str(info.get("kind")) == "literal"),
    }
    return _inject_graph_theme(html, node_info=node_info, graph_stats=graph_stats, heading=heading)


def _inject_graph_theme(
    html: str,
    node_info: Optional[Dict[str, Dict[str, object]]] = None,
    graph_stats: Optional[Dict[str, int]] = None,
    heading: str = "Graph View",
) -> str:
    safe_node_info = json.dumps(node_info or {}, ensure_ascii=False).replace("</", "<\\/")
    safe_graph_stats = json.dumps(graph_stats or {}, ensure_ascii=False).replace("</", "<\\/")
    safe_heading = json.dumps(heading, ensure_ascii=False).replace("</", "<\\/")
    theme = """
    <style>
      html, body {
        background: #f3f6f8 !important;
        color: #17262b !important;
        margin: 0;
        font-family: Arial, sans-serif;
      }
      #mynetwork {
        border: 0 !important;
        border-radius: 0 !important;
        background: #eef3f6 !important;
        width: 100% !important;
        min-width: 520px !important;
      }
      #kgGraphShell {
        display: flex;
        gap: 0;
        height: 100%;
        width: 100%;
        border: 1px solid #cbd5dc;
        border-radius: 6px;
        overflow: hidden;
        background: #eef3f6;
      }
      #kgCanvasPane {
        background: #eef3f6;
        flex: 1 1 auto;
        min-width: 520px;
        position: relative;
      }
      #kgGraphToolbar {
        align-items: center;
        background: rgba(255,255,255,0.86);
        border-bottom: 1px solid #d7e0e7;
        box-sizing: border-box;
        color: #14213d;
        display: flex;
        gap: 8px;
        height: 38px;
        padding: 6px 10px;
      }
      #kgGraphToolbar strong {
        font-size: 13px;
        margin-right: 8px;
      }
      #kgGraphToolbar button {
        background: #ffffff;
        border: 1px solid #b7c6d1;
        border-radius: 3px;
        color: #14213d;
        cursor: pointer;
        font-size: 12px;
        padding: 4px 8px;
      }
      #kgGraphToolbar button:hover {
        border-color: #0067b1;
      }
      #kgBottomBar {
        align-items: center;
        background: rgba(255,255,255,0.9);
        border-top: 1px solid #d7e0e7;
        bottom: 0;
        box-sizing: border-box;
        display: flex;
        gap: 10px;
        left: 0;
        padding: 6px 10px;
        position: absolute;
        right: 0;
        z-index: 5;
      }
      #kgSearch {
        border: 1px solid #b7c6d1;
        border-radius: 3px;
        font-size: 12px;
        padding: 5px 8px;
        width: 190px;
      }
      #kgNodePanel {
        background: #172331;
        border-left: 1px solid #243344;
        border-radius: 0;
        box-sizing: border-box;
        color: #dbe7ee;
        font-size: 13px;
        line-height: 1.45;
        overflow: auto;
        padding: 0;
        width: 310px;
      }
      #kgNodePanel h3 {
        color: #ffffff;
        font-size: 20px;
        font-weight: 500;
        line-height: 1.08;
        margin: 0;
      }
      #kgNodePanel h4 {
        color: #dbe7ee;
        font-size: 15px;
        margin: 0 0 8px;
      }
      #kgNodePanel .muted {
        color: #8ea3b1;
      }
      #kgNodePanel .badge {
        background: #233446;
        border: 1px solid #355069;
        border-radius: 999px;
        color: #9bc8ff;
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        margin: 0 4px 6px 0;
        padding: 3px 8px;
      }
      #kgNodePanel .section {
        border-top: 1px solid #2a3c50;
        padding: 13px 16px;
      }
      #kgNodePanel .panel-head {
        background: #101927;
        padding: 16px;
      }
      #kgNodePanel .panel-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin: 5px 0;
      }
      #kgNodePanel .panel-value {
        color: #ffffff;
        font-weight: 700;
        text-align: right;
      }
      #kgNodePanel ul {
        margin: 6px 0 0 18px;
        padding: 0;
      }
      #kgNodePanel li {
        margin: 3px 0;
      }
      #kgNodePanel code {
        background: #101927;
        border-radius: 4px;
        color: #bfe4ff;
        display: block;
        overflow-wrap: anywhere;
        padding: 6px;
        white-space: normal;
      }
      @media (max-width: 820px) {
        #kgGraphShell { flex-direction: column; }
        #mynetwork, #kgCanvasPane { width: 100% !important; min-width: 0 !important; }
        #kgNodePanel { width: 100%; max-height: 220px; }
      }
    </style>
    <script>
      const KG_NODE_INFO = __KG_NODE_INFO__;
      const KG_GRAPH_STATS = __KG_GRAPH_STATS__;
      const KG_GRAPH_HEADING = __KG_GRAPH_HEADING__;
      function kgEscape(value) {
        return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
          '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
        }[ch]));
      }
      function kgList(items, formatter) {
        if (!items || !items.length) return '<p class="muted">No information available.</p>';
        return '<ul>' + items.slice(0, 12).map(formatter).join('') + '</ul>';
      }
      function kgRenderNodePanel(nodeId) {
        const panel = document.getElementById('kgNodePanel');
        if (!panel) return;
        const info = KG_NODE_INFO[nodeId];
        if (!info) {
          kgRenderDefaultPanel('<p class="muted">No information available for this selection.</p>');
          return;
        }
        const labels = info.labels && info.labels.length ? info.labels : [info.label];
        const comments = info.comments || [];
        const types = info.types || [];
        const typeBadges = types.map((t) => '<span class="badge">' + kgEscape(t) + '</span>').join('');
        const description = comments.length
          ? comments.map((c) => '<p>' + kgEscape(c) + '</p>').join('')
          : '<p class="muted">No description available.</p>';
        panel.innerHTML = kgPanelHeader() +
          '<div class="section"><h4>Selection Details</h4><p><strong>Name:</strong> <em>' + kgEscape(labels[0] || info.label) + '</em></p>' +
          '<p><strong>Type:</strong> ' + kgEscape(info.kind || 'node') + '</p>' +
          typeBadges +
          '<code>' + kgEscape(info.uri || info.id) + '</code></div>' +
          '<div class="section"><h4>Description</h4>' + description + '</div>' +
          '<div class="section"><h4>Outgoing relationships</h4>' +
            kgList(info.outgoing, (r) => '<li><strong>' + kgEscape(r.predicate) + '</strong> &rarr; ' + kgEscape(r.target) + '</li>') +
          '</div>' +
          '<div class="section"><h4>Incoming relationships</h4>' +
            kgList(info.incoming, (r) => '<li>' + kgEscape(r.source) + ' &rarr; <strong>' + kgEscape(r.predicate) + '</strong></li>') +
          '</div>';
      }
      function kgPanelHeader() {
        return '<div class="panel-head"><h3>True Demand KG<br>Ontology View</h3>' +
          '<p class="muted">' + kgEscape(KG_GRAPH_HEADING || 'Ontology visualization') + '</p></div>' +
          '<div class="section"><h4>Statistics</h4>' +
          '<div class="panel-row"><span>Nodes</span><span class="panel-value">' + kgEscape(KG_GRAPH_STATS.nodes || 0) + '</span></div>' +
          '<div class="panel-row"><span>Relationships</span><span class="panel-value">' + kgEscape(KG_GRAPH_STATS.edges || 0) + '</span></div>' +
          '<div class="panel-row"><span>Classes/entities</span><span class="panel-value">' + kgEscape(KG_GRAPH_STATS.classes || 0) + '</span></div>' +
          '</div>';
      }
      function kgRenderDefaultPanel(extraHtml) {
        const panel = document.getElementById('kgNodePanel');
        if (!panel) return;
        panel.innerHTML = kgPanelHeader() +
          '<div class="section"><h4>Description</h4><p>Interactive ontology-style view of graph relationships used by the True Demand KG QA system.</p></div>' +
          '<div class="section"><h4>Selection Details</h4>' +
          (extraHtml || '<p class="muted">Click a class or node to inspect details available in this view.</p>') +
          '</div>';
      }
      function kgSetGraphLabelScale(scale) {
        if (typeof edges === 'undefined' || typeof nodes === 'undefined') return;
        let edgeSize = 9;
        let edgeColor = '#2f5f91';
        let edgeStroke = 2;
        let edgeOpacity = 0.58;
        let edgeBackground = 'rgba(230,241,255,0.9)';
        let nodeSize = 12;
        let nodeStroke = 4;
        let nodeColor = '#17262b';
        if (scale < 0.58) {
          edgeSize = 0;
          edgeColor = 'rgba(47,95,145,0)';
          edgeStroke = 0;
          edgeOpacity = 0.22;
          edgeBackground = 'rgba(230,241,255,0)';
          nodeSize = 0;
          nodeStroke = 0;
          nodeColor = 'rgba(23,38,43,0)';
        } else if (scale < 0.82) {
          edgeSize = 0;
          edgeColor = 'rgba(47,95,145,0)';
          edgeStroke = 0;
          edgeOpacity = 0.34;
          edgeBackground = 'rgba(230,241,255,0)';
          nodeSize = 10;
          nodeStroke = 3;
        } else if (scale > 1.75) {
          edgeSize = 15;
          edgeColor = '#063f6a';
          edgeStroke = 4;
          edgeOpacity = 0.92;
          edgeBackground = 'rgba(255,255,255,0.98)';
          nodeSize = 19;
          nodeStroke = 6;
        } else if (scale > 1.25) {
          edgeSize = 12;
          edgeColor = '#0a4f82';
          edgeStroke = 3;
          edgeOpacity = 0.82;
          edgeBackground = 'rgba(255,255,255,0.96)';
          nodeSize = 16;
          nodeStroke = 5;
        }
        const edgeUpdates = edges.getIds().map((id) => ({
          id,
          color: { color: '#7d8995', highlight: '#ff4b4b', hover: '#0067b1', opacity: edgeOpacity },
          font: {
            size: edgeSize,
            align: 'middle',
            color: edgeColor,
            background: edgeBackground,
            strokeWidth: edgeStroke,
            strokeColor: '#ffffff'
          }
        }));
        const nodeUpdates = nodes.getIds().map((id) => ({
          id,
          font: {
            size: nodeSize,
            color: nodeColor,
            face: 'Arial',
            vadjust: scale > 1.25 ? -2 : 0,
            strokeWidth: nodeStroke,
            strokeColor: '#ffffff'
          }
        }));
        edges.update(edgeUpdates);
        nodes.update(nodeUpdates);
      }
      function kgDisablePhysics() {
        if (typeof network === 'undefined') return;
        if (typeof network.stopSimulation === 'function') {
          network.stopSimulation();
        }
        network.setOptions({
          physics: {
            enabled: false,
            stabilization: false
          }
        });
      }
      function kgFreezeGraphExcept(nodeIds) {
        if (typeof nodes === 'undefined' || typeof network === 'undefined') return;
        const movable = new Set(nodeIds || []);
        const positions = network.getPositions();
        const updates = nodes.getIds().map((id) => {
          const pos = positions[id] || {};
          return {
            id,
            x: Number.isFinite(pos.x) ? pos.x : undefined,
            y: Number.isFinite(pos.y) ? pos.y : undefined,
            fixed: { x: !movable.has(id), y: !movable.has(id) }
          };
        });
        nodes.update(updates);
      }
      let kgDragState = null;
      function kgPointerCanvas(params) {
        return params && params.pointer && params.pointer.canvas ? params.pointer.canvas : null;
      }
      function kgMoveDirectNeighbors(centerId, dx, dy, strength) {
        if (typeof network === 'undefined' || typeof nodes === 'undefined') return;
        if (!Number.isFinite(dx) || !Number.isFinite(dy)) return;
        const positions = network.getPositions();
        const updates = [];
        (network.getConnectedNodes(centerId) || []).slice(0, 12).forEach((id) => {
          const pos = positions[id];
          if (!pos) return;
          updates.push({
            id,
            x: pos.x + (dx * strength),
            y: pos.y + (dy * strength),
            fixed: { x: true, y: true }
          });
        });
        if (updates.length) nodes.update(updates);
      }
      function kgRelaxConnectedNodes(centerId) {
        if (typeof network === 'undefined' || typeof nodes === 'undefined') return;
        const positions = network.getPositions();
        const center = positions[centerId];
        if (!center) return;
        const updates = [];
        (network.getConnectedNodes(centerId) || []).slice(0, 10).forEach((id) => {
          const pos = positions[id];
          if (!pos) return;
          updates.push({
            id,
            x: pos.x + ((center.x - pos.x) * 0.035),
            y: pos.y + ((center.y - pos.y) * 0.035),
            fixed: { x: true, y: true }
          });
        });
        if (updates.length) nodes.update(updates);
      }
      function kgInstallNodePanel(attempts) {
        const networkEl = document.getElementById('mynetwork');
        if (!networkEl || document.getElementById('kgGraphShell')) return;
        const shell = document.createElement('div');
        shell.id = 'kgGraphShell';
        networkEl.parentNode.insertBefore(shell, networkEl);
        const canvasPane = document.createElement('div');
        canvasPane.id = 'kgCanvasPane';
        const toolbar = document.createElement('div');
        toolbar.id = 'kgGraphToolbar';
        toolbar.innerHTML = '<strong>' + kgEscape(KG_GRAPH_HEADING || 'Ontology view') + '</strong>' +
          '<button type="button" id="kgFitBtn">Fit</button>' +
          '<button type="button" id="kgZoomInBtn">+</button>' +
          '<button type="button" id="kgZoomOutBtn">-</button>' +
          '<button type="button" id="kgPauseBtn">Pause</button>';
        const bottom = document.createElement('div');
        bottom.id = 'kgBottomBar';
        bottom.innerHTML = '<input id="kgSearch" type="search" placeholder="Search class or entity" />' +
          '<span class="muted">Zoom, drag nodes, click to inspect details.</span>';
        shell.appendChild(canvasPane);
        canvasPane.appendChild(toolbar);
        canvasPane.appendChild(networkEl);
        canvasPane.appendChild(bottom);
        const panel = document.createElement('aside');
        panel.id = 'kgNodePanel';
        shell.appendChild(panel);
        kgRenderDefaultPanel();
        if (typeof network === 'undefined') {
          if ((attempts || 0) < 20) window.setTimeout(() => kgInstallNodePanel((attempts || 0) + 1), 100);
          return;
        }
        document.getElementById('kgFitBtn').addEventListener('click', () => network.fit({animation: true}));
        document.getElementById('kgZoomInBtn').addEventListener('click', () => network.moveTo({scale: network.getScale() * 1.18}));
        document.getElementById('kgZoomOutBtn').addEventListener('click', () => network.moveTo({scale: network.getScale() * 0.85}));
        document.getElementById('kgPauseBtn').addEventListener('click', () => kgDisablePhysics());
        document.getElementById('kgSearch').addEventListener('input', (event) => {
          const term = String(event.target.value || '').trim().toLowerCase();
          if (!term || term.length < 2) return;
          const match = nodes.getIds().find((id) => {
            const info = KG_NODE_INFO[id] || {};
            return String(info.label || id).toLowerCase().includes(term);
          });
          if (match) {
            network.selectNodes([match]);
            network.focus(match, {scale: Math.max(network.getScale(), 1.1), animation: true});
            kgRenderNodePanel(match);
          }
        });
        network.on('click', (params) => {
          if (params.nodes && params.nodes.length) {
            kgRenderNodePanel(params.nodes[0]);
            network.selectNodes([params.nodes[0]]);
          } else {
            kgRenderDefaultPanel();
          }
        });
        network.once('stabilizationIterationsDone', () => {
          kgFreezeGraphExcept([]);
          kgDisablePhysics();
          kgSetGraphLabelScale(network.getScale());
        });
        network.once('stabilized', () => {
          kgFreezeGraphExcept([]);
          kgDisablePhysics();
          kgSetGraphLabelScale(network.getScale());
        });
        window.setTimeout(() => {
          kgFreezeGraphExcept([]);
          kgDisablePhysics();
          kgSetGraphLabelScale(network.getScale());
        }, 1800);
        network.on('zoom', (params) => {
          kgSetGraphLabelScale(params.scale || network.getScale());
        });
        network.on('dragStart', (params) => {
          if (params.nodes && params.nodes.length) {
            const selectedId = params.nodes[0];
            const pointer = kgPointerCanvas(params);
            kgDragState = {
              selectedId,
              x: pointer ? pointer.x : null,
              y: pointer ? pointer.y : null
            };
            kgFreezeGraphExcept([selectedId]);
            kgDisablePhysics();
          }
        });
        network.on('dragging', (params) => {
          if (!kgDragState || !kgDragState.selectedId) return;
          const pointer = kgPointerCanvas(params);
          if (!pointer || kgDragState.x === null || kgDragState.y === null) return;
          const dx = pointer.x - kgDragState.x;
          const dy = pointer.y - kgDragState.y;
          kgDragState.x = pointer.x;
          kgDragState.y = pointer.y;
          kgMoveDirectNeighbors(kgDragState.selectedId, dx, dy, 0.18);
        });
        network.on('dragEnd', (params) => {
          if (params.nodes && params.nodes.length) {
            const selectedId = params.nodes[0];
            kgRelaxConnectedNodes(selectedId);
            kgFreezeGraphExcept([]);
            kgDisablePhysics();
            network.selectNodes([selectedId]);
          }
          kgDragState = null;
        });
      }
      window.addEventListener('load', () => kgInstallNodePanel(0));
    </script>
    """
    theme = theme.replace("__KG_NODE_INFO__", safe_node_info)
    theme = theme.replace("__KG_GRAPH_STATS__", safe_graph_stats)
    theme = theme.replace("__KG_GRAPH_HEADING__", safe_heading)
    return html.replace("</body>", theme + "</body>")


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
    node_details: Dict[str, Dict[str, object]] = {}
    for term in node_terms:
        node_details[_node_id(term)] = {
            "label": _short_term(term),
            "uri": str(term),
            "kind": _node_kind(term),
            "types": [],
            "labels": [],
            "comments": [],
            "outgoing": [],
            "incoming": [],
        }

    for s, p, o in edges:
        if s not in positions or o not in positions:
            continue
        sid = _node_id(s)
        oid = _node_id(o)
        pred_short = _short_term(p)
        if str(p) == str(RDFS.label):
            node_details.setdefault(sid, {}).setdefault("labels", []).append(str(o))
        elif str(p) == str(RDFS.comment):
            node_details.setdefault(sid, {}).setdefault("comments", []).append(str(o))
        elif str(p) == str(RDF.type):
            node_details.setdefault(sid, {}).setdefault("types", []).append(_short_term(o))
        else:
            node_details.setdefault(sid, {}).setdefault("outgoing", []).append({"predicate": pred_short, "target": _short_term(o)})
            node_details.setdefault(oid, {}).setdefault("incoming", []).append({"predicate": pred_short, "source": _short_term(s)})
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
            f"<text class='edgeLabel' x='{mx:.1f}' y='{my - 4:.1f}' fill='#91a4a4' font-size='10' "
            f"text-anchor='middle'>{escape(_short_term(p))}</text>"
        )

    node_markup = []
    for idx, (term, (x, y)) in enumerate(positions.items()):
        is_entity = _is_entity(term)
        fill = "#b9f2f2" if is_entity else "#54656d"
        stroke = "#7de4df" if is_entity else "#70838b"
        label = escape(_node_visual_label(term))
        node_id = escape(_node_id(term))
        node_markup.append(
            f"<g class='graphNode' data-node-id='{node_id}' style='cursor:pointer'>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='23' fill='{fill}' stroke='{stroke}' "
            "stroke-width='2.4' opacity='0.95' />"
            f"<text class='nodeLabel' x='{x:.1f}' y='{y + 34 + ((idx % 3) * 9):.1f}' fill='#edf4f3' font-size='12' "
            f"text-anchor='middle' paint-order='stroke' stroke='#0b151a' stroke-width='4'>{label}</text>"
            "</g>"
        )

    safe_node_details = json.dumps(node_details, ensure_ascii=False).replace("</", "<\\/")
    return (
        "<style>"
        "html,body{margin:0;background:#101719;color:#edf4f3;font-family:Arial,sans-serif;}"
        "h4{margin:0 0 12px;color:#edf4f3;font-family:Arial,sans-serif;}"
        ".toolbar{display:flex;gap:8px;margin-bottom:12px;}"
        ".toolbar button{background:#172124;color:#edf4f3;border:1px solid #26373a;"
        "border-radius:6px;padding:7px 10px;cursor:pointer;}"
        ".toolbar button:hover{border-color:#19d6c6;}"
        ".svgShell{display:flex;border:1px solid #26373a;border-radius:14px;overflow:hidden;background:#0f1f28;}"
        "#graphWrap{flex:1;min-width:520px;overflow:hidden;background:radial-gradient(circle at 50% 35%, #123342 0%, #0f1f28 58%, #0b151a 100%);cursor:grab;}"
        "#svgNodePanel{width:310px;background:#172331;border-left:1px solid #243344;color:#dbe7ee;overflow:auto;font-size:13px;line-height:1.45;}"
        "#svgNodePanel .head{background:#101927;padding:16px;}#svgNodePanel h3{margin:0;color:#fff;font-size:20px;line-height:1.1;}#svgNodePanel h4{margin:0 0 8px;color:#dbe7ee;}"
        "#svgNodePanel .section{border-top:1px solid #2a3c50;padding:13px 16px;}#svgNodePanel .muted{color:#8ea3b1;}#svgNodePanel code{display:block;white-space:normal;overflow-wrap:anywhere;background:#101927;color:#bfe4ff;border-radius:4px;padding:6px;}#svgNodePanel li{margin:3px 0;}"
        "@media(max-width:820px){.svgShell{flex-direction:column;}#graphWrap{min-width:0;}#svgNodePanel{width:100%;max-height:240px;}}"
        "</style>"
        f"<h4>{escape(heading)}</h4>"
        "<div class='toolbar'>"
        "<button onclick='zoomGraph(1.2)'>Zoom in</button>"
        "<button onclick='zoomGraph(0.83)'>Zoom out</button>"
        "<button onclick='resetGraph()'>Reset</button>"
        "</div>"
        "<div class='svgShell'><div id='graphWrap'>"
        f"<svg id='graphSvg' viewBox='0 0 {width} {height}' width='100%' height='{height}' "
        "xmlns='http://www.w3.org/2000/svg' role='img'>"
        "<defs><marker id='arrow' markerWidth='8' markerHeight='8' refX='7' refY='3' "
        "orient='auto'><path d='M0,0 L0,6 L8,3 z' fill='#5f7477'/></marker></defs>"
        "<g id='graphViewport'>"
        + "".join(edge_markup)
        + "".join(edge_labels)
        + "".join(node_markup)
        + "</g></svg></div><aside id='svgNodePanel'></aside></div>"
        "<script>"
        f"const NODE_INFO={safe_node_details};"
        "const svg=document.getElementById('graphSvg');"
        "const viewport=document.getElementById('graphViewport');"
        "const wrap=document.getElementById('graphWrap');"
        "const panel=document.getElementById('svgNodePanel');"
        "let scale=1, tx=0, ty=0, dragging=false, sx=0, sy=0;"
        "function esc(v){return String(v??'').replace(/[&<>\"']/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#039;'}[c]));}"
        "function list(items,fmt){return items&&items.length?'<ul>'+items.slice(0,12).map(fmt).join('')+'</ul>':'<p class=\"muted\">No information available.</p>';}"
        "function defaultPanel(){panel.innerHTML='<div class=\"head\"><h3>True Demand KG<br>Graph View</h3><p class=\"muted\">Click a node to inspect available details.</p></div><div class=\"section\"><h4>Statistics</h4><p>Nodes: "+str(len(node_terms))+"<br>Relationships: "+str(len(edges))+"</p></div>';}"
        "function renderNode(id){const info=NODE_INFO[id];if(!info){defaultPanel();return;}const label=(info.labels&&info.labels.length?info.labels[0]:info.label);const desc=(info.comments&&info.comments.length)?info.comments.map(c=>'<p>'+esc(c)+'</p>').join(''):'<p class=\"muted\">No description available.</p>';panel.innerHTML='<div class=\"head\"><h3>'+esc(label)+'</h3><p class=\"muted\">'+esc(info.kind||'node')+'</p></div><div class=\"section\"><h4>Identifier</h4><code>'+esc(info.uri||id)+'</code></div><div class=\"section\"><h4>Description</h4>'+desc+'</div><div class=\"section\"><h4>Outgoing relationships</h4>'+list(info.outgoing,r=>'<li><strong>'+esc(r.predicate)+'</strong> → '+esc(r.target)+'</li>')+'</div><div class=\"section\"><h4>Incoming relationships</h4>'+list(info.incoming,r=>'<li>'+esc(r.source)+' → <strong>'+esc(r.predicate)+'</strong></li>')+'</div>';}"
        "function updateLabelVisibility(){document.querySelectorAll('.edgeLabel').forEach(e=>e.style.display=scale>1.15?'block':'none');document.querySelectorAll('.nodeLabel').forEach(e=>{e.style.display=scale<0.55?'none':'block';e.setAttribute('font-size',scale>1.45?'14':'12');});}"
        "function applyGraph(){viewport.setAttribute('transform',`translate(${tx} ${ty}) scale(${scale})`);updateLabelVisibility();}"
        "function zoomGraph(f){scale=Math.max(0.25,Math.min(5,scale*f));applyGraph();}"
        "function resetGraph(){scale=1;tx=0;ty=0;applyGraph();}"
        "wrap.addEventListener('wheel',(e)=>{e.preventDefault();zoomGraph(e.deltaY<0?1.1:0.9);},{passive:false});"
        "wrap.addEventListener('mousedown',(e)=>{if(e.target.closest('.graphNode'))return;dragging=true;sx=e.clientX;sy=e.clientY;wrap.style.cursor='grabbing';});"
        "window.addEventListener('mouseup',()=>{dragging=false;wrap.style.cursor='grab';});"
        "window.addEventListener('mousemove',(e)=>{if(!dragging)return;tx+=(e.clientX-sx)/scale;ty+=(e.clientY-sy)/scale;sx=e.clientX;sy=e.clientY;applyGraph();});"
        "document.querySelectorAll('.graphNode').forEach(g=>g.addEventListener('click',(e)=>{e.stopPropagation();renderNode(g.dataset.nodeId);}));"
        "defaultPanel();updateLabelVisibility();"
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
    use_limit = int(limit) > 0
    try:
        total_rows = list(graph.query("SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"))
        total = int(total_rows[0][0].toPython()) if total_rows else 0
        limit_clause = f" LIMIT {int(limit)}" if use_limit else ""
        rows = graph.query(f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }}{limit_clause}")
        return [(row[0], row[1], row[2]) for row in rows], total
    except Exception:
        total = 0
        out: List[Tuple] = []
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
