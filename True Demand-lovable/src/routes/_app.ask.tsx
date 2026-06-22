import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Clock,
  HelpCircle,
  Loader2,
  Network,
  Send,
  Sparkles,
  BookOpen,
  ListChecks,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";
import { AutocompleteInput } from "@/components/AutocompleteInput";
import { CytoscapeGraph } from "@/components/CytoscapeGraph";
import { TypeBadge } from "@/components/TypeBadge";
import { api } from "@/lib/api";
import type {
  Capability,
  DecisionKind,
  Example,
  GraphNode,
  GraphPayload,
  QuestionResponse,
} from "@/lib/types";

export const Route = createFileRoute("/_app/ask")({
  head: () => ({ meta: [{ title: "Ask — True Demand KG QA" }] }),
  component: AskPage,
});

type DecMeta = { label: string; color: string; bg: string };
function decisionMeta(d: DecisionKind): DecMeta {
  switch (d) {
    case "direct":
      return {
        label: "Direct Answer",
        color: "var(--success)",
        bg: "color-mix(in oklab, var(--success) 14%, white)",
      };
    case "auto":
      return {
        label: "Auto Answer",
        color: "var(--primary)",
        bg: "color-mix(in oklab, var(--primary) 12%, white)",
      };
    case "clarification":
      return {
        label: "Needs Clarification",
        color: "var(--warning-foreground)",
        bg: "color-mix(in oklab, var(--warning) 35%, white)",
      };
    case "unsupported":
      return {
        label: "Unsupported",
        color: "var(--destructive)",
        bg: "color-mix(in oklab, var(--destructive) 12%, white)",
      };
  }
}

function isAnswerable(r: QuestionResponse): boolean {
  return r.decision !== "clarification" && r.decision !== "unsupported" && !r.unsupported;
}

function AskPage() {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState<QuestionResponse | null>(null);
  const [evidence, setEvidence] = useState<GraphPayload | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [caps, setCaps] = useState<Capability[]>([]);
  const [showDev, setShowDev] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);
  const [requestError, setRequestError] = useState("");
  const [activeGuide, setActiveGuide] = useState<"examples" | "builder" | "topics">("examples");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.examples().then(setExamples);
    api.capabilities().then(setCaps);
  }, []);

  async function submit(question: string) {
    if (!question.trim() || loading) return;
    setLoading(true);
    setFeedback(null);
    setResp(null);
    setEvidence(null);
    setSelectedNode(null);
    setShowEvidence(false);
    setShowDev(false);
    setRequestError("");
    try {
      const r = await api.ask(question.trim());
      setResp(r);
      if (isAnswerable(r)) {
        const ev = await api.evidence(r.caseId);
        setEvidence(ev);
      }
      queueMicrotask(() =>
        answerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
      );
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "The KGQA request failed.");
    } finally {
      setLoading(false);
    }
  }

  async function chooseInterpretation(id: string) {
    if (!resp) return;
    setLoading(true);
    setSelectedNode(null);
    try {
      const r = await api.clarify(resp.caseId, id);
      setResp(r);
      if (isAnswerable(r)) {
        const ev = await api.evidence(r.caseId);
        setEvidence(ev);
      } else {
        setEvidence(null);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="px-6 py-6 max-w-[1200px] mx-auto">
      <div className="mb-4">
        <h1 className="text-[20px] font-semibold tracking-tight">Ask the Knowledge Graph</h1>
        <p className="text-[13px] text-muted-foreground mt-0.5">
          Type a natural-language question. The graph schema autocompletes as you type.
        </p>
      </div>

      <div className="space-y-3">
        <AutocompleteInput
          value={q}
          onChange={setQ}
          onSubmit={() => submit(q)}
          disabled={loading}
        />
        <div className="flex items-center gap-2">
          <button
            onClick={() => submit(q)}
            disabled={loading || !q.trim()}
            className="inline-flex items-center gap-1.5 rounded-md bg-primary text-primary-foreground text-[13px] font-medium px-3.5 py-1.5 hover:opacity-90 disabled:opacity-50 transition-opacity"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Send className="h-3.5 w-3.5" />
            )}
            {loading ? "Asking…" : "Ask"}
          </button>
          <span className="text-[11px] text-muted-foreground">Press ⌘+Enter to submit</span>
        </div>
      </div>

      {/* Question guide */}
      <div className="mt-5 rounded-md border border-border bg-card">
        <div className="flex border-b border-border text-[12px]">
          {[
            { k: "examples" as const, label: "Examples", icon: Sparkles },
            { k: "builder" as const, label: "Guided Builder", icon: ListChecks },
            { k: "topics" as const, label: "Available Topics", icon: BookOpen },
          ].map(({ k, label, icon: Icon }) => (
            <button
              key={k}
              onClick={() => setActiveGuide(k)}
              className="flex items-center gap-1.5 px-3.5 py-2 font-medium transition-colors"
              style={{
                color: activeGuide === k ? "var(--primary)" : "var(--muted-foreground)",
                borderBottom:
                  activeGuide === k ? "2px solid var(--primary)" : "2px solid transparent",
                marginBottom: -1,
              }}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </div>
        <div className="p-3">
          {activeGuide === "examples" && (
            <div className="flex flex-wrap gap-1.5">
              {examples.map((ex) => (
                <button
                  key={ex.id}
                  onClick={() => {
                    setQ(ex.text);
                    submit(ex.text);
                  }}
                  className="text-[12px] px-2.5 py-1 rounded-full border border-border bg-muted hover:bg-accent text-foreground/80 transition-colors"
                >
                  {ex.text}
                </button>
              ))}
            </div>
          )}
          {activeGuide === "builder" && (
            <GuidedBuilder
              capabilities={caps}
              onCompose={(text) => setQ(text)}
              onAsk={(text) => {
                setQ(text);
                submit(text);
              }}
            />
          )}
          {activeGuide === "topics" && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              {caps.map((c) => (
                <div key={c.family} className="rounded border border-border p-2.5">
                  <div className="flex items-center justify-between">
                    <div className="text-[13px] font-medium">{c.family}</div>
                    <div className="text-[10px] text-muted-foreground">{c.templates} templates</div>
                  </div>
                  <div className="text-[11px] text-muted-foreground mt-0.5">{c.description}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {requestError && (
        <div className="mt-4 rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-[12px] text-destructive">
          {requestError}
        </div>
      )}

      {/* Response area */}
      {resp && (
        <div ref={answerRef} className="mt-6 space-y-4">
          {resp.decision === "clarification" && (
            <ClarificationCard resp={resp} onSelect={chooseInterpretation} loading={loading} />
          )}

          {(resp.decision === "unsupported" || resp.unsupported) && (
            <UnsupportedCard
              resp={resp}
              onRevise={() => {
                setResp(null);
                setEvidence(null);
              }}
            />
          )}

          {isAnswerable(resp) && (
            <>
              <AnswerCard resp={resp} />

              {/* Evidence + Dev toggle row */}
              <div className="flex flex-wrap items-center gap-2">
                {evidence && evidence.nodes.length > 0 && (
                  <button
                    onClick={() => setShowEvidence((v) => !v)}
                    className="inline-flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-[12px] font-medium hover:bg-accent transition-colors"
                    aria-expanded={showEvidence}
                  >
                    <Network className="h-3.5 w-3.5 text-primary" />
                    {showEvidence ? "Hide evidence graph" : "View evidence graph"}
                    <span className="text-[11px] text-muted-foreground ml-1">
                      {evidence.nodes.length} nodes · {evidence.edges.length} relations
                    </span>
                    {showEvidence ? (
                      <ChevronDown className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5" />
                    )}
                  </button>
                )}
                <div className="ml-auto flex items-center gap-3 text-[12px] text-muted-foreground">
                  <span className="inline-flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {resp.responseTimeMs} ms
                  </span>
                  <span>Confidence {(resp.confidence * 100).toFixed(0)}%</span>
                  <span>Entropy {resp.entropy.toFixed(2)}</span>
                  <span className="text-[11px] text-muted-foreground mr-1">Helpful?</span>
                  <button
                    onClick={() => setFeedback("up")}
                    className="p-1 rounded hover:bg-accent transition-colors"
                    style={{
                      color: feedback === "up" ? "var(--success)" : "var(--muted-foreground)",
                    }}
                    aria-label="Thumbs up"
                  >
                    <ThumbsUp className="h-3.5 w-3.5" />
                  </button>
                  <button
                    onClick={() => setFeedback("down")}
                    className="p-1 rounded hover:bg-accent transition-colors"
                    style={{
                      color: feedback === "down" ? "var(--destructive)" : "var(--muted-foreground)",
                    }}
                    aria-label="Thumbs down"
                  >
                    <ThumbsDown className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              {/* Evidence graph + side panel — only when toggled */}
              {showEvidence && evidence && evidence.nodes.length > 0 && (
                <div className="rounded-md border border-border bg-card">
                  <div className="px-3.5 py-2 border-b border-border flex items-center justify-between">
                    <div>
                      <div className="text-[13px] font-medium">Answer Evidence Graph</div>
                      <div className="text-[11px] text-muted-foreground">
                        Question-specific subgraph · {evidence.nodes.length} nodes /{" "}
                        {evidence.edges.length} relations bound by the SPARQL query · Click a node
                        for details
                      </div>
                    </div>
                    <GraphLegend />
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px]">
                    <div className="p-2">
                      <CytoscapeGraph
                        data={evidence}
                        height={620}
                        layout="cose"
                        highlightIds={evidence.pathNodeIds ?? evidence.nodes.map((n) => n.id)}
                        highlightEdgeIds={evidence.pathEdgeIds}
                        onNodeClick={setSelectedNode}
                        detailsPanel={
                          <NodeDetailsPanel
                            node={selectedNode}
                            onClose={() => setSelectedNode(null)}
                          />
                        }
                      />
                    </div>
                    <NodeDetailsPanel node={selectedNode} onClose={() => setSelectedNode(null)} />
                  </div>
                </div>
              )}
            </>
          )}

          {/* Developer details (always available when there's a response) */}
          <div className="rounded-md border border-border bg-card">
            <button
              onClick={() => setShowDev((v) => !v)}
              className="w-full px-3.5 py-2 flex items-center gap-1.5 text-[12px] font-medium text-muted-foreground hover:bg-muted/50 transition-colors"
            >
              {showDev ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
              Developer Details
              <span className="ml-auto text-[10px] uppercase tracking-wide">
                {resp.diagnostics.family}
                {resp.diagnostics.template ? ` · ${resp.diagnostics.template}` : ""}
              </span>
            </button>
            {showDev && (
              <div className="border-t border-border p-3.5 space-y-3">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[12px]">
                  <KV label="Case ID" value={resp.caseId} mono />
                  <KV label="Family" value={resp.diagnostics.family} />
                  <KV label="Template" value={resp.diagnostics.template ?? "—"} mono />
                  <KV
                    label="Ranker Score"
                    value={resp.diagnostics.rankerScore?.toFixed(2) ?? "—"}
                  />
                </div>
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground mb-1">
                    SPARQL
                  </div>
                  <pre className="rounded bg-muted text-[12px] p-3 overflow-auto font-mono leading-relaxed">
                    {resp.sparql || "— no query produced —"}
                  </pre>
                </div>
                {resp.diagnostics.safetyFlags && resp.diagnostics.safetyFlags.length > 0 && (
                  <div className="text-[12px]">
                    Safety flags: {resp.diagnostics.safetyFlags.join(", ")}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function KV({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={mono ? "font-mono text-[12px] mt-0.5 break-all" : "text-[12px] mt-0.5"}>
        {value}
      </div>
    </div>
  );
}

function AnswerCard({ resp }: { resp: QuestionResponse }) {
  const d = decisionMeta(resp.decision);
  return (
    <div className="rounded-lg border border-border bg-card shadow-sm">
      <div className="px-4 py-2.5 border-b border-border flex items-center gap-2">
        <Sparkles className="h-3.5 w-3.5 text-primary" />
        <span className="text-[12px] font-medium text-foreground">Answer</span>
        <span
          className="type-badge"
          style={{ backgroundColor: d.bg, color: d.color, borderColor: "transparent" }}
        >
          {d.label}
        </span>
        <div className="ml-auto text-[11px] text-muted-foreground">
          Case <span className="font-mono">{resp.caseId}</span>
        </div>
      </div>
      <div className="px-5 py-4">
        <p className="text-[15px] leading-relaxed text-foreground">{resp.answer}</p>
        {resp.table && (
          <div className="mt-4 overflow-auto rounded border border-border">
            <table className="w-full text-[12px]">
              <thead className="bg-muted/60">
                <tr>
                  {resp.table.columns.map((c) => (
                    <th
                      key={c.key}
                      className="px-3 py-1.5 text-left font-medium text-muted-foreground"
                    >
                      {c.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {resp.table.rows.map((row, i) => (
                  <tr key={i} className="border-t border-border">
                    {resp.table!.columns.map((c) => (
                      <td key={c.key} className="px-3 py-1.5">
                        {String(row[c.key] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function ClarificationCard({
  resp,
  onSelect,
  loading,
}: {
  resp: QuestionResponse;
  onSelect: (id: string) => void;
  loading: boolean;
}) {
  const d = decisionMeta("clarification");
  const items = (resp.interpretations ?? []).slice(0, 3);
  return (
    <div className="rounded-md border border-border bg-card">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <span
          className="type-badge"
          style={{ backgroundColor: d.bg, color: d.color, borderColor: "transparent" }}
        >
          {d.label}
        </span>
        <div className="text-[12px] text-muted-foreground inline-flex items-center gap-1.5">
          <HelpCircle className="h-3.5 w-3.5" />
          The question maps to multiple distinct readings. Pick the intended one to continue.
        </div>
        <div className="ml-auto text-[11px] text-muted-foreground">
          Case <span className="font-mono">{resp.caseId}</span>
        </div>
      </div>
      <div className="p-2 space-y-1">
        {items.map((it) => (
          <button
            key={it.id}
            disabled={loading}
            onClick={() => onSelect(it.id)}
            className="w-full text-left px-3 py-2.5 rounded border border-transparent hover:border-border hover:bg-accent/60 transition-colors flex items-start gap-3 disabled:opacity-60"
          >
            <div className="flex-1">
              <div className="text-[13px] text-foreground">{it.text}</div>
              {it.hint && (
                <div className="text-[11px] font-mono text-muted-foreground mt-1 truncate">
                  {it.hint}
                </div>
              )}
            </div>
            <span className="text-[11px] text-muted-foreground shrink-0 mt-0.5">
              {(it.confidence * 100).toFixed(0)}%
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function UnsupportedCard({ resp, onRevise }: { resp: QuestionResponse; onRevise: () => void }) {
  const d = decisionMeta("unsupported");
  return (
    <div className="rounded-md border border-border bg-card">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <span
          className="type-badge"
          style={{ backgroundColor: d.bg, color: d.color, borderColor: "transparent" }}
        >
          {d.label}
        </span>
        <div className="ml-auto text-[11px] text-muted-foreground">
          Case <span className="font-mono">{resp.caseId}</span>
        </div>
      </div>
      <div className="px-4 py-4 flex items-start gap-3">
        <AlertTriangle className="h-5 w-5 mt-0.5 text-amber-600 shrink-0" aria-hidden />
        <div className="flex-1 space-y-2">
          <div className="text-[13px] font-medium">No supporting evidence in the graph</div>
          <p className="text-[12px] text-muted-foreground leading-relaxed">
            The query produced no rows or graph evidence. We are not presenting this as a
            high-confidence answer. Try narrowing the question — name a specific metric, product,
            region or scope — or pick an example below.
          </p>
          <div className="flex items-center gap-2 pt-1">
            <button
              onClick={onRevise}
              className="text-[12px] px-3 py-1.5 rounded-md border border-border hover:bg-accent transition-colors"
            >
              Revise question
            </button>
            <span className="text-[11px] text-muted-foreground">
              Confidence {(resp.confidence * 100).toFixed(0)}% · Entropy {resp.entropy.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function GraphLegend() {
  const items: { type: GraphNode["type"] }[] = [
    { type: "Class" },
    { type: "ObjectProperty" },
    { type: "DatatypeProperty" },
    { type: "Datatype" },
    { type: "Entity" },
    { type: "Literal" },
  ];
  return (
    <div className="hidden md:flex items-center gap-1 flex-wrap max-w-[420px] justify-end">
      {items.map((i) => (
        <TypeBadge key={i.type} type={i.type} />
      ))}
    </div>
  );
}

function NodeDetailsPanel({ node, onClose }: { node: GraphNode | null; onClose: () => void }) {
  return (
    <aside className="border-l border-border bg-muted/30 p-3 min-h-[420px]">
      {!node ? (
        <div className="text-[12px] text-muted-foreground h-full flex items-center justify-center text-center px-2">
          Click any node in the graph to inspect its IRI, type and definition.
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <TypeBadge type={node.type} />
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-accent"
              aria-label="Close panel"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <div>
            <div className="text-[13px] font-semibold leading-tight break-words">
              {node.type === "Literal" ? `"${node.label}"` : node.label}
            </div>
            {node.iri && (
              <div className="text-[11px] font-mono text-muted-foreground mt-0.5 break-all">
                {node.iri}
              </div>
            )}
          </div>
          {node.type === "Literal" && (
            <KV label="Datatype" value={node.datatype ?? "xsd:string"} mono />
          )}
          {node.definition && (
            <div>
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-0.5">
                Definition
              </div>
              <p className="text-[12px] leading-relaxed text-foreground/80">{node.definition}</p>
            </div>
          )}
          {node.properties && node.properties.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
                Properties
              </div>
              <dl className="space-y-1">
                {node.properties.map((p) => (
                  <div key={p.key} className="text-[12px] grid grid-cols-[90px_1fr] gap-2">
                    <dt className="text-muted-foreground">{p.key}</dt>
                    <dd className="font-mono break-all">{p.value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

function GuidedBuilder({
  capabilities,
  onCompose,
  onAsk,
}: {
  capabilities: Capability[];
  onCompose: (text: string) => void;
  onAsk: (text: string) => void;
}) {
  const [family, setFamily] = useState("");
  const [dimension, setDimension] = useState("");
  const selectedCapability = capabilities.find((capability) => capability.family === family);
  const composed = family && dimension ? `Show ${family.toLowerCase()} by ${dimension}.` : "";

  return (
    <div className="space-y-3">
      <p className="text-[11px] text-muted-foreground">
        Every available combination is backed by an executable capability query that returned graph
        rows.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        <label className="text-[11px] text-muted-foreground">
          Metric / capability
          <select
            value={family}
            onChange={(event) => {
              setFamily(event.target.value);
              setDimension("");
            }}
            className="mt-1 w-full rounded border border-input bg-card px-2.5 py-2 text-[12px] text-foreground"
          >
            <option value="">Select a capability</option>
            {capabilities.map((capability) => (
              <option key={capability.family} value={capability.family}>
                {capability.family}
              </option>
            ))}
          </select>
        </label>
        <label className="text-[11px] text-muted-foreground">
          Breakdown
          <select
            value={dimension}
            disabled={!selectedCapability}
            onChange={(event) => setDimension(event.target.value)}
            className="mt-1 w-full rounded border border-input bg-card px-2.5 py-2 text-[12px] text-foreground disabled:opacity-50"
          >
            <option value="">Select a breakdown</option>
            {(selectedCapability?.dimensions ?? []).map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>

      {composed && (
        <div className="flex flex-wrap items-center gap-2 rounded border border-border bg-muted/40 px-3 py-2">
          <div className="min-w-0 flex-1 text-[12px]">
            <span className="text-muted-foreground mr-1.5">Generated question:</span>
            <span className="text-foreground">{composed}</span>
          </div>
          <button
            onClick={() => onCompose(composed)}
            className="text-[11px] px-2 py-1 rounded border border-border hover:bg-accent"
          >
            Use as question
          </button>
          <button
            onClick={() => onAsk(composed)}
            className="inline-flex items-center gap-1 text-[11px] px-2.5 py-1 rounded bg-primary text-primary-foreground hover:opacity-90"
          >
            <Send className="h-3 w-3" /> Ask
          </button>
        </div>
      )}
    </div>
  );
}
