import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { CaseSummary, ConfidenceMetrics, DecisionKind } from "@/lib/types";

export const Route = createFileRoute("/_app/confidence")({
  head: () => ({ meta: [{ title: "Confidence Dashboard — True Demand KG QA" }] }),
  component: ConfidencePage,
});

function ConfidencePage() {
  const [m, setM] = useState<ConfidenceMetrics | null>(null);
  const [family, setFamily] = useState<string>("all");
  const [decision, setDecision] = useState<DecisionKind | "all">("all");
  const [correctness, setCorrectness] = useState<"all" | "correct" | "incorrect" | "unknown">(
    "all",
  );
  const [minConf, setMinConf] = useState(0);

  useEffect(() => {
    api.metrics().then(setM);
  }, []);

  const families = useMemo(() => (m ? Array.from(new Set(m.cases.map((c) => c.family))) : []), [m]);
  const cases = useMemo(() => {
    if (!m) return [];
    return m.cases.filter(
      (c) =>
        (family === "all" || c.family === family) &&
        (decision === "all" || c.decision === decision) &&
        (correctness === "all" ||
          (correctness === "correct" && c.correct === true) ||
          (correctness === "incorrect" && c.correct === false) ||
          (correctness === "unknown" && c.correct === null)) &&
        c.confidence >= minConf,
    );
  }, [m, family, decision, correctness, minConf]);

  if (!m) return <div className="p-6 text-[13px] text-muted-foreground">Loading metrics…</div>;

  return (
    <div className="px-6 py-6 max-w-[1280px] mx-auto space-y-5">
      <div>
        <h1 className="text-[20px] font-semibold tracking-tight">Confidence Dashboard</h1>
        <p className="text-[13px] text-muted-foreground mt-0.5">
          System accuracy, coverage and routing summaries.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-3">
        <Stat label="System Accuracy" value={pct(m.systemAccuracy ?? 0)} accent="success" />
        <Stat label="Direct Accuracy" value={pct(m.directAccuracy ?? 0)} accent="success" />
        <Stat label="LLM Accuracy" value={pct(m.llmAccuracy ?? 0)} />
        <Stat label="Forced Top-1" value={pct(m.forcedTop1)} />
        <Stat label="Any-Correct" value={pct(m.anyCorrect)} />
        <Stat label="Auto-Answer Accuracy" value={pct(m.autoAnswerAccuracy)} accent="success" />
        <Stat label="Coverage" value={pct(m.coverage)} accent="primary" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="Confidence Buckets">
          <div className="space-y-2 text-[12px]">
            {m.confidenceBuckets.map((b) => {
              const acc = b.correct / b.count;
              return (
                <div key={b.bucket}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono">{b.bucket}</span>
                    <span className="text-muted-foreground">
                      {b.correct}/{b.count} · {pct(acc)}
                    </span>
                  </div>
                  <div className="h-1.5 rounded bg-muted overflow-hidden">
                    <div
                      className="h-full"
                      style={{ width: `${acc * 100}%`, background: "var(--primary)" }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <Card title="Entropy Routing">
          <div className="space-y-2 text-[12px]">
            {m.entropyRouting.map((r) => {
              const total = m.entropyRouting.reduce((s, x) => s + x.count, 0);
              const w = r.count / total;
              return (
                <div key={r.route}>
                  <div className="flex items-center justify-between mb-1">
                    <span>{r.route}</span>
                    <span className="text-muted-foreground">
                      {r.count} · {pct(w)}
                    </span>
                  </div>
                  <div className="h-1.5 rounded bg-muted overflow-hidden">
                    <div
                      className="h-full"
                      style={{ width: `${w * 100}%`, background: "var(--chart-2)" }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      <Card title="Accuracy vs Coverage">
        <AccuracyCoverageChart points={m.accuracyVsCoverage} />
      </Card>

      <Card title="Case Browser">
        <div className="flex flex-wrap gap-2 mb-3 text-[12px]">
          <Select
            label="Family"
            value={family}
            onChange={setFamily}
            options={[{ v: "all", l: "All" }, ...families.map((f) => ({ v: f, l: f }))]}
          />
          <Select
            label="Decision"
            value={decision}
            onChange={(v) => setDecision(v as DecisionKind | "all")}
            options={[
              { v: "all", l: "All" },
              { v: "direct", l: "Direct" },
              { v: "auto", l: "Auto" },
              { v: "clarification", l: "Clarification" },
            ]}
          />
          <Select
            label="Correctness"
            value={correctness}
            onChange={(v) => setCorrectness(v as "all" | "correct" | "incorrect" | "unknown")}
            options={[
              { v: "all", l: "All" },
              { v: "correct", l: "Correct" },
              { v: "incorrect", l: "Incorrect" },
              { v: "unknown", l: "Unknown" },
            ]}
          />
          <label className="inline-flex items-center gap-1.5">
            <span className="text-muted-foreground">Min conf</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={minConf}
              onChange={(e) => setMinConf(Number(e.target.value))}
              className="w-28"
            />
            <span className="font-mono w-10">{minConf.toFixed(2)}</span>
          </label>
        </div>
        <div className="overflow-auto rounded border border-border">
          <table className="w-full text-[12px]">
            <thead className="bg-muted/60">
              <tr className="text-left text-muted-foreground">
                <th className="px-3 py-1.5 font-medium">Case</th>
                <th className="px-3 py-1.5 font-medium">Question</th>
                <th className="px-3 py-1.5 font-medium">Family</th>
                <th className="px-3 py-1.5 font-medium">Decision</th>
                <th className="px-3 py-1.5 font-medium text-right">Conf</th>
                <th className="px-3 py-1.5 font-medium text-right">Entropy</th>
                <th className="px-3 py-1.5 font-medium">Result</th>
                <th className="px-3 py-1.5 font-medium">Flags</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c) => (
                <CaseRow key={c.id} c={c} />
              ))}
              {cases.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-3 py-6 text-center text-muted-foreground">
                    No cases match filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function CaseRow({ c }: { c: CaseSummary }) {
  const dColor =
    c.decision === "direct"
      ? "var(--success)"
      : c.decision === "auto"
        ? "var(--primary)"
        : "var(--warning-foreground)";
  const resultLabel = c.correct === true ? "✓ correct" : c.correct === false ? "✗ incorrect" : "—";
  const resultColor =
    c.correct === true
      ? "var(--success)"
      : c.correct === false
        ? "var(--destructive)"
        : "var(--muted-foreground)";
  return (
    <tr className="border-t border-border">
      <td className="px-3 py-1.5 font-mono text-[11px]">{c.id}</td>
      <td className="px-3 py-1.5 truncate max-w-[280px]">{c.question}</td>
      <td className="px-3 py-1.5">{c.family}</td>
      <td className="px-3 py-1.5" style={{ color: dColor }}>
        {c.decision}
      </td>
      <td className="px-3 py-1.5 text-right font-mono">{c.confidence.toFixed(2)}</td>
      <td className="px-3 py-1.5 text-right font-mono">{c.entropy.toFixed(2)}</td>
      <td className="px-3 py-1.5" style={{ color: resultColor }}>
        {resultLabel}
      </td>
      <td className="px-3 py-1.5 text-muted-foreground">{c.safetyFlags.join(", ") || "—"}</td>
    </tr>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "primary" | "success";
}) {
  const color =
    accent === "success"
      ? "var(--success)"
      : accent === "primary"
        ? "var(--primary)"
        : "var(--foreground)";
  return (
    <div className="rounded-md border border-border bg-card p-3.5">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-[22px] font-semibold tabular-nums" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border bg-card">
      <div className="px-3.5 py-2 border-b border-border text-[12px] font-medium">{title}</div>
      <div className="p-3.5">{children}</div>
    </div>
  );
}

function Select<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: { v: T; l: string }[];
}) {
  return (
    <label className="inline-flex items-center gap-1.5">
      <span className="text-muted-foreground">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="rounded border border-input bg-card px-2 py-1 text-[12px]"
      >
        {options.map((o) => (
          <option key={o.v} value={o.v}>
            {o.l}
          </option>
        ))}
      </select>
    </label>
  );
}

function AccuracyCoverageChart({ points }: { points: { coverage: number; accuracy: number }[] }) {
  const W = 600,
    H = 200,
    P = 28;
  const x = (v: number) => P + v * (W - P * 2);
  const y = (v: number) => H - P - (v - 0.5) * (H - P * 2) * 2;
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.coverage)},${y(p.accuracy)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-48">
      <rect x={P} y={P} width={W - P * 2} height={H - P * 2} fill="none" stroke="var(--border)" />
      {[0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((t) => (
        <g key={t}>
          <line
            x1={P}
            x2={W - P}
            y1={y(t)}
            y2={y(t)}
            stroke="var(--border)"
            strokeDasharray="2 3"
          />
          <text x={4} y={y(t) + 3} fontSize={9} fill="var(--muted-foreground)">
            {(t * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      <path d={path} fill="none" stroke="var(--primary)" strokeWidth={1.5} />
      {points.map((p, i) => (
        <circle key={i} cx={x(p.coverage)} cy={y(p.accuracy)} r={2.5} fill="var(--primary)" />
      ))}
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--muted-foreground)">
        Coverage →
      </text>
      <text x={10} y={P - 8} fontSize={10} fill="var(--muted-foreground)">
        Accuracy
      </text>
    </svg>
  );
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}
