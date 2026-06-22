import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { Maximize2, Minimize2 } from "lucide-react";
import type { EntityType, GraphNode, GraphPayload } from "@/lib/types";

/* WebOWL-style force-directed graph using Cytoscape.
   - circular/pill nodes colored by type
   - curved parallel edges with relationship labels
   - solid black arrows
   - drag, zoom, pan, fullscreen w/ optional side details panel */

const TYPE_COLOR: Partial<Record<EntityType, string>> = {
  Class: "#0066B3",
  ObjectProperty: "#64748B",
  DatatypeProperty: "#B7791F",
  Property: "#64748B",
  Datatype: "#94A3B8",
  Metric: "#0EA5E9",
  Dimension: "#8B5CF6",
  Scope: "#F59E0B",
  Entity: "#475569",
  Literal: "#A16207",
};

function colorFor(t: EntityType) {
  return TYPE_COLOR[t] ?? "#475569";
}

interface Props {
  data: GraphPayload;
  highlightIds?: string[];
  highlightEdgeIds?: string[];
  onNodeClick?: (node: GraphNode) => void;
  height?: number | string;
  detailsPanel?: ReactNode;
}

function GraphCanvas({
  data,
  highlightIds,
  highlightEdgeIds,
  onNodeClick,
}: Pick<Props, "data" | "highlightIds" | "highlightEdgeIds" | "onNodeClick">) {
  const ref = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  const elements = useMemo<ElementDefinition[]>(() => {
    const nodes: ElementDefinition[] = data.nodes.map((n) => ({
      data: {
        id: n.id,
        label: n.label,
        type: n.type,
        color: colorFor(n.type),
        ref: n,
      },
    }));
    const edges: ElementDefinition[] = data.edges.map((e) => ({
      data: { id: e.id, source: e.source, target: e.target, label: e.label },
    }));
    return [...nodes, ...edges];
  }, [data]);

  useEffect(() => {
    if (!ref.current) return;
    const hlN = new Set(highlightIds ?? []);
    const hlE = new Set(highlightEdgeIds ?? []);

    const cy = cytoscape({
      container: ref.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            "border-width": 2,
            "border-color": "#ffffff",
            label: "data(label)",
            color: "#0f172a",
            "font-size": 11,
            "font-family": "Inter, system-ui, sans-serif",
            "font-weight": 600,
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": 6,
            "text-wrap": "wrap",
            "text-max-width": "140px",
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.85,
            "text-background-padding": "2px",
            "text-background-shape": "roundrectangle",
            width: 38,
            height: 38,
            shape: "ellipse",
            "overlay-opacity": 0,
          },
        },
        {
          selector: "node:selected",
          style: { "border-color": "#E11D48", "border-width": 3 },
        },
        {
          selector: "node.highlight",
          style: { "border-color": "#E11D48", "border-width": 3 },
        },
        {
          selector: "node.faded",
          style: { opacity: 0.25 },
        },
        {
          selector: "edge",
          style: {
            "curve-style": "bezier",
            "control-point-step-size": 50,
            width: 1.4,
            "line-color": "#0f172a",
            "target-arrow-color": "#0f172a",
            "target-arrow-shape": "triangle",
            "arrow-scale": 1.1,
            label: "data(label)",
            "font-size": 9,
            "font-family": "Geist Mono, ui-monospace, monospace",
            color: "#0f172a",
            "text-rotation": "autorotate" as unknown as undefined,
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.92,
            "text-background-padding": "2px",
            "text-background-shape": "roundrectangle",
            "text-border-color": "#e2e8f0",
            "text-border-width": 1,
            "text-border-opacity": 1,
          },
        },
        {
          selector: "edge.highlight",
          style: {
            "line-color": "#0066B3",
            "target-arrow-color": "#0066B3",
            width: 2.2,
            color: "#0066B3",
            "font-weight": 700,
          },
        },
        {
          selector: "edge.faded",
          style: { opacity: 0.15 },
        },
      ],
      layout: {
        name: "cose",
        animate: false,
        fit: true,
        nodeRepulsion: () => 9000,
        idealEdgeLength: () => 130,
        edgeElasticity: () => 120,
        gravity: 0.25,
        numIter: 1500,
        padding: 40,
      } as cytoscape.LayoutOptions,
    });

    // Re-run layout once the container has real dimensions, then fit.
    const runFit = () => {
      try {
        cy.resize();
        cy.layout({
          name: "cose",
          animate: false,
          fit: true,
          padding: 40,
        } as cytoscape.LayoutOptions).run();
        cy.fit(undefined, 40);
      } catch {
        // The graph can be briefly detached while the route changes.
      }
    };
    const ro = new ResizeObserver(() => runFit());
    if (ref.current) ro.observe(ref.current);
    // Initial deferred fit (container may be 0 height at first paint)
    const t = setTimeout(runFit, 60);

    // Apply highlights
    if (hlN.size || hlE.size) {
      cy.elements().addClass("faded");
      cy.nodes().forEach((n) => {
        if (hlN.has(n.id())) {
          n.removeClass("faded").addClass("highlight");
        }
      });
      cy.edges().forEach((e) => {
        const id = e.id();
        const connected = hlN.has(e.source().id()) && hlN.has(e.target().id());
        if (hlE.has(id) || connected) e.removeClass("faded").addClass("highlight");
      });
    }

    cy.on("tap", "node", (evt) => {
      const n = evt.target;
      const ref = n.data("ref") as GraphNode | undefined;
      if (ref && onNodeClick) onNodeClick(ref);
    });

    cyRef.current = cy;
    return () => {
      clearTimeout(t);
      ro.disconnect();
      cy.destroy();
      cyRef.current = null;
    };
  }, [elements, highlightIds, highlightEdgeIds, onNodeClick]);

  return (
    <div
      ref={ref}
      style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", minHeight: 1 }}
    />
  );
}

export function WebOwlGraph(props: Props) {
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

  return (
    <>
      <div
        className="relative w-full rounded-md border border-border bg-card overflow-hidden"
        style={{ height }}
      >
        {toolbar}
        <GraphCanvas
          data={props.data}
          highlightIds={props.highlightIds}
          highlightEdgeIds={props.highlightEdgeIds}
          onNodeClick={props.onNodeClick}
        />
      </div>

      {fullscreen && (
        <div className="fixed inset-0 z-50 bg-background">
          <div className="relative h-full w-full grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="relative h-full w-full">
              {toolbar}
              <GraphCanvas
                data={props.data}
                highlightIds={props.highlightIds}
                highlightEdgeIds={props.highlightEdgeIds}
                onNodeClick={props.onNodeClick}
              />
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
