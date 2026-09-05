import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import { WebOwlGraph } from "@/components/WebOwlGraph";
import { TypeBadge } from "@/components/TypeBadge";
import { api } from "@/lib/api";
import type { EntityType, GraphNode, GraphPayload } from "@/lib/types";

export const Route = createFileRoute("/_app/explorer")({
  head: () => ({ meta: [{ title: "Knowledge Graph Explorer — True Demand KG QA" }] }),
  component: ExplorerPage,
});

const ALL_TYPES: EntityType[] = [
  "Class",
  "ObjectProperty",
  "DatatypeProperty",
  "Property",
  "Datatype",
  "Metric",
  "Dimension",
  "Scope",
  "Entity",
  "Literal",
];

const DATA_TYPES = new Set<EntityType>([
  "Entity",
  "Literal",
  "Class",
  "ObjectProperty",
  "DatatypeProperty",
  "Property",
  "Datatype",
]);
const ONTOLOGY_TYPES = new Set<EntityType>([
  "Class",
  "ObjectProperty",
  "DatatypeProperty",
  "Property",
  "Datatype",
  "Metric",
  "Dimension",
  "Scope",
  "Entity",
]);

function ExplorerPage() {
  const [data, setData] = useState<GraphPayload | null>(null);
  const [view, setView] = useState<"ontology" | "data">("ontology");
  const [query, setQuery] = useState("");
  const [types, setTypes] = useState<Set<EntityType>>(new Set(ALL_TYPES));
  const [selected, setSelected] = useState<GraphNode | null>(null);

  useEffect(() => {
    setData(null);
    setSelected(null);
    const request = view === "ontology" ? api.ontology() : api.dataGraph();
    request.then(setData);
  }, [view]);

  const filtered = useMemo<GraphPayload | null>(() => {
    if (!data) return null;
    const allowedClasses = view === "ontology" ? ONTOLOGY_TYPES : DATA_TYPES;
    const nodes = data.nodes.filter(
      (n) =>
        types.has(n.type) &&
        allowedClasses.has(n.type) &&
        (!query || n.label.toLowerCase().includes(query.toLowerCase())),
    );
    const ids = new Set(nodes.map((n) => n.id));
    const edges = data.edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    return { nodes, edges };
  }, [data, view, query, types]);

  const relationships = useMemo(() => {
    if (!selected || !data) return [];
    return data.edges
      .filter((e) => e.source === selected.id || e.target === selected.id)
      .map((e) => ({
        ...e,
        other: e.source === selected.id ? e.target : e.source,
        direction: e.source === selected.id ? "out" : ("in" as const),
      }));
  }, [selected, data]);

  const detailsBody = selected ? (
    <div className="p-4">
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-[15px] font-semibold leading-tight">{selected.label}</div>
          <div className="mt-1">
            <TypeBadge type={selected.type} />
          </div>
        </div>
        <button onClick={() => setSelected(null)} className="p-1 rounded hover:bg-accent">
          <X className="h-3.5 w-3.5 text-muted-foreground" />
        </button>
      </div>
      {selected.description && (
        <p className="text-[12px] text-muted-foreground leading-relaxed mb-3">
          {selected.description}
        </p>
      )}
      <div className="mb-4">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">
          Identifier
        </div>
        <div className="font-mono text-[11px] bg-muted px-2 py-1 rounded">{selected.id}</div>
      </div>
      <div className="mb-4">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">
          Properties
        </div>
        {selected.properties && selected.properties.length > 0 ? (
          <div className="space-y-1 text-[12px]">
            {selected.properties.map((p) => (
              <div key={p.key} className="flex">
                <span className="text-muted-foreground w-1/2">{p.key}</span>
                <span className="flex-1">{p.value}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-muted-foreground">No properties defined.</div>
        )}
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5">
          Relationships
        </div>
        {relationships.length === 0 ? (
          <div className="text-[11px] text-muted-foreground">No connections.</div>
        ) : (
          <ul className="space-y-1">
            {relationships.map((r) => (
              <li key={r.id} className="text-[12px] flex items-center gap-1.5">
                <span className="text-muted-foreground">{r.direction === "out" ? "→" : "←"}</span>
                <span className="font-mono text-[11px] text-primary">{r.label}</span>
                <span className="text-muted-foreground">·</span>
                <span className="truncate">{r.other}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  ) : null;

  return (
    <div className="flex h-[calc(100vh-3rem)]">
      <div className="flex-1 flex flex-col p-4 gap-3 min-w-0">
        <div className="flex items-center gap-2">
          <h1 className="text-[18px] font-semibold tracking-tight mr-2">
            Knowledge Graph Explorer
          </h1>
          <div className="inline-flex rounded-md border border-border bg-card p-0.5 text-[12px]">
            {(["ontology", "data"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className="px-2.5 py-1 rounded transition-colors"
                style={{
                  backgroundColor: view === v ? "var(--primary)" : "transparent",
                  color: view === v ? "var(--primary-foreground)" : "var(--muted-foreground)",
                }}
              >
                {v === "ontology" ? "Ontology" : "Data"}
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search nodes"
                className="pl-7 pr-2 py-1.5 w-52 text-[12px] rounded-md border border-input bg-card outline-none focus:border-ring focus:ring-2 focus:ring-ring/20"
              />
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] text-muted-foreground mr-1">Filter:</span>
          {ALL_TYPES.map((t) => (
            <button
              key={t}
              onClick={() =>
                setTypes((s) => {
                  const next = new Set(s);
                  if (next.has(t)) next.delete(t);
                  else next.add(t);
                  return next;
                })
              }
              className="text-[11px] px-2 py-0.5 rounded-full border transition-colors"
              style={{
                borderColor: "var(--border)",
                backgroundColor: types.has(t) ? "var(--accent)" : "transparent",
                color: types.has(t) ? "var(--accent-foreground)" : "var(--muted-foreground)",
              }}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="flex-1 min-h-0">
          {filtered && (
            <WebOwlGraph
              data={filtered}
              height="100%"
              onNodeClick={setSelected}
              detailsPanel={detailsBody}
            />
          )}
        </div>
      </div>

      {/* Details panel */}
      <aside
        className="border-l border-border bg-card overflow-auto transition-all"
        style={{ width: selected ? 340 : 0 }}
      >
        {detailsBody}
      </aside>
    </div>
  );
}
