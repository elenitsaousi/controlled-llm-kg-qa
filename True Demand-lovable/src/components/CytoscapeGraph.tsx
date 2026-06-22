import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  Handle,
  Position,
  useEdgesState,
  useNodesState,
  type Node,
  type Edge,
  type NodeProps,
  type NodeMouseHandler,
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";
import dagre from "dagre";
import {
  Box,
  Braces,
  Database,
  FileText,
  Hash,
  Link2,
  Maximize2,
  Minimize2,
  Ruler,
  Tag,
  Type,
} from "lucide-react";
import type { EntityType, GraphNode, GraphPayload } from "@/lib/types";

/* ---------------- Node visual presentation ---------------- */

type Meta = {
  tag: string;
  icon: React.ComponentType<{ className?: string }>;
  tone: string; // tailwind classes for bg/text/border
  ring: string;
};

const META: Partial<Record<EntityType, Meta>> = {
  Class: {
    tag: "Class",
    icon: Box,
    tone: "bg-primary/10 text-primary border-primary/30",
    ring: "ring-primary",
  },
  ObjectProperty: {
    tag: "Object property",
    icon: Link2,
    tone: "bg-slate-100 text-slate-950 border-slate-400",
    ring: "ring-slate-500",
  },
  DatatypeProperty: {
    tag: "Datatype property",
    icon: Braces,
    tone: "bg-amber-50 text-amber-950 border-amber-300",
    ring: "ring-amber-500",
  },
  Property: {
    tag: "Property",
    icon: Tag,
    tone: "bg-sky-50 text-sky-950 border-sky-300",
    ring: "ring-sky-400",
  },
  Datatype: {
    tag: "Datatype",
    icon: Type,
    tone: "bg-slate-50 text-slate-800 border-slate-300",
    ring: "ring-slate-400",
  },
  Metric: {
    tag: "Metric",
    icon: Hash,
    tone: "bg-emerald-50 text-emerald-950 border-emerald-300",
    ring: "ring-emerald-400",
  },
  Dimension: {
    tag: "Dimension",
    icon: Ruler,
    tone: "bg-violet-50 text-violet-900 border-violet-300",
    ring: "ring-violet-400",
  },
  Scope: {
    tag: "Scope",
    icon: Braces,
    tone: "bg-amber-50 text-amber-950 border-amber-200",
    ring: "ring-amber-400",
  },
  Entity: {
    tag: "Entity",
    icon: Database,
    tone: "bg-card text-foreground border-border",
    ring: "ring-foreground/20",
  },
  Literal: {
    tag: "Literal",
    icon: Type,
    tone: "bg-muted text-foreground border-border",
    ring: "ring-muted-foreground",
  },
};

const FALLBACK_META: Meta = {
  tag: "Node",
  icon: FileText,
  tone: "bg-card text-foreground border-border",
  ring: "ring-foreground/20",
};

function metaFor(type: EntityType): Meta {
  return META[type] ?? FALLBACK_META;
}

/* ---------------- Card node ---------------- */

type CardData = { node: GraphNode; selected: boolean };

function CardNode({ data }: NodeProps<Node<CardData>>) {
  const n = data.node;
  const meta = metaFor(n.type);
  const Icon = meta.icon;
  const label = n.type === "Literal" ? `"${n.label}"` : n.label;
  return (
    <div
      className={[
        "group rounded-xl border px-3 py-2 shadow-sm transition",
        "min-w-[200px] max-w-[260px]",
        meta.tone,
        data.selected ? `ring-2 ${meta.ring}` : "ring-0",
      ].join(" ")}
    >
      <Handle type="target" position={Position.Left} className="!opacity-0" />
      <div className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide opacity-75">
        <Icon className="h-3 w-3" />
        {meta.tag}
      </div>
      <div className="mt-0.5 text-[13px] font-semibold leading-snug line-clamp-3" title={label}>
        {label}
      </div>
      {n.definition && (
        <div className="mt-0.5 text-[11px] opacity-80 line-clamp-2">{n.definition}</div>
      )}
      <Handle type="source" position={Position.Right} className="!opacity-0" />
    </div>
  );
}

const NODE_TYPES = { card: CardNode };

/* ---------------- Layout ---------------- */

const NODE_W = 260;
const NODE_H = 130;

function layoutNodes(nodes: GraphNode[], edges: { source: string; target: string }[]) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: "LR",
    nodesep: 110,
    ranksep: 200,
    marginx: 32,
    marginy: 32,
    edgesep: 60,
    ranker: "tight-tree",
  });

  g.setDefaultEdgeLabel(() => ({}));
  const ids = new Set(nodes.map((n) => n.id));
  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of edges) {
    if (ids.has(e.source) && ids.has(e.target)) g.setEdge(e.source, e.target);
  }
  dagre.layout(g);
  return nodes.map((n) => {
    const p = g.node(n.id);
    return { n, x: (p?.x ?? 0) - NODE_W / 2, y: (p?.y ?? 0) - NODE_H / 2 };
  });
}

/* ---------------- Component ---------------- */

interface Props {
  data: GraphPayload;
  highlightIds?: string[];
  highlightEdgeIds?: string[];
  onNodeClick?: (node: GraphNode) => void;
  height?: number | string;
  layout?: "cose" | "concentric" | "breadthfirst";
  /** Optional details panel rendered to the right of the graph when fullscreen is active. */
  detailsPanel?: ReactNode;
}

function FlowInner({
  data,
  highlightIds,
  highlightEdgeIds,
  onNodeClick,
}: Pick<Props, "data" | "highlightIds" | "highlightEdgeIds" | "onNodeClick">) {
  const hlNodes = useMemo(() => new Set(highlightIds ?? []), [highlightIds]);
  const hlEdges = useMemo(() => {
    if (highlightEdgeIds?.length) return new Set(highlightEdgeIds);
    if (!highlightIds?.length) return new Set<string>();
    const s = new Set<string>();
    for (const e of data.edges) {
      if (hlNodes.has(e.source) && hlNodes.has(e.target)) s.add(e.id);
    }
    return s;
  }, [highlightEdgeIds, highlightIds, data.edges, hlNodes]);

  const [selectedId, setSelectedId] = useState<string | null>(null);

  const initial = useMemo(() => {
    const positioned = layoutNodes(data.nodes, data.edges);
    const nodes: Node<CardData>[] = positioned.map(({ n, x, y }) => ({
      id: n.id,
      type: "card",
      position: { x, y },
      data: { node: n, selected: false },
      draggable: true,
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    }));
    const edges: Edge[] = data.edges.map((e) => {
      const on = hlEdges.has(e.id);
      const stroke = on ? "var(--primary)" : "var(--foreground)";
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.label,
        type: "smoothstep",
        labelBgPadding: [4, 2] as [number, number],
        labelBgBorderRadius: 3,
        labelBgStyle: { fill: "var(--card)", fillOpacity: 0.92 },
        labelStyle: {
          fontSize: 10,
          fontFamily: "Geist Mono, ui-monospace, monospace",
          fill: on ? "var(--primary)" : "var(--foreground)",
          letterSpacing: "0.04em",
          fontWeight: on ? 600 : 500,
        },
        style: { stroke, strokeWidth: on ? 2 : 1.5, strokeOpacity: on ? 1 : 0.75 },
        markerEnd: { type: "arrowclosed" as const, color: stroke, width: 18, height: 18 },
      };
    });

    return { nodes, edges };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const [rfNodes, setRfNodes, onNodesChange] = useNodesState(initial.nodes);
  const [rfEdges, , onEdgesChange] = useEdgesState(initial.edges);

  // Reset positions only when the underlying graph payload changes.
  useEffect(() => {
    setRfNodes(initial.nodes);
    setSelectedId(null);
  }, [initial, setRfNodes]);

  // Apply selection ring without losing drag positions.
  const styledNodes = useMemo(
    () =>
      rfNodes.map((n) => ({
        ...n,
        data: { ...(n.data as CardData), selected: selectedId === n.id },
      })),
    [rfNodes, selectedId],
  );

  const handleClick = useCallback<NodeMouseHandler>(
    (_e, node) => {
      const d = node.data as CardData;
      setSelectedId(d.node.id);
      onNodeClick?.(d.node);
    },
    [onNodeClick],
  );

  return (
    <ReactFlow
      nodes={styledNodes}
      edges={rfEdges}
      nodeTypes={NODE_TYPES}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={handleClick}
      onPaneClick={() => setSelectedId(null)}
      nodesDraggable
      nodesConnectable={false}
      elementsSelectable
      panOnDrag
      zoomOnScroll
      fitView
      fitViewOptions={{ padding: 0.18 }}
      minZoom={0.3}
      maxZoom={1.8}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={18} size={1} color="var(--border)" />
      <Controls
        showInteractive={false}
        className="!border-border !bg-card !shadow-none [&>button]:!border-border [&>button]:!bg-card [&>button]:!text-foreground"
      />
    </ReactFlow>
  );
}

export function CytoscapeGraph(props: Props) {
  const { height = 480, detailsPanel } = props;
  const [fullscreen, setFullscreen] = useState(false);

  const toolbar = (
    <button
      type="button"
      onClick={() => setFullscreen((v) => !v)}
      className="absolute right-2 top-2 z-10 inline-flex items-center gap-1 rounded-md border border-border bg-card px-2 py-1 text-[11px] text-foreground shadow-sm hover:bg-accent"
      aria-label={fullscreen ? "Exit fullscreen" : "Fullscreen"}
    >
      {fullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
      {fullscreen ? "Exit" : "Full"}
    </button>
  );

  const graph = (
    <ReactFlowProvider>
      <FlowInner
        data={props.data}
        highlightIds={props.highlightIds}
        highlightEdgeIds={props.highlightEdgeIds}
        onNodeClick={props.onNodeClick}
      />
    </ReactFlowProvider>
  );

  return (
    <>
      {/* Inline view — stays mounted even in fullscreen so exiting restores state. */}
      <div
        className="relative w-full rounded-md border border-border bg-card overflow-hidden"
        style={{ height }}
      >
        {toolbar}
        {graph}
      </div>

      {/* Fullscreen overlay with optional side panel. */}
      {fullscreen && (
        <div className="fixed inset-0 z-50 bg-background">
          <div className="relative h-full w-full grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="relative h-full w-full">
              {toolbar}
              <ReactFlowProvider>
                <FlowInner
                  data={props.data}
                  highlightIds={props.highlightIds}
                  highlightEdgeIds={props.highlightEdgeIds}
                  onNodeClick={props.onNodeClick}
                />
              </ReactFlowProvider>
            </div>
            {detailsPanel !== undefined && (
              <div className="h-full overflow-auto border-l border-border bg-card">
                {detailsPanel}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
