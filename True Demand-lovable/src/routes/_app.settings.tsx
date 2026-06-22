import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { CheckCircle2, AlertTriangle, XCircle, Loader2 } from "lucide-react";
import { api, getApiBaseUrl, getUseMocks, setApiBaseUrl, setUseMocks } from "@/lib/api";
import type { HealthStatus } from "@/lib/types";

export const Route = createFileRoute("/_app/settings")({
  head: () => ({ meta: [{ title: "Settings — True Demand KG QA" }] }),
  component: SettingsPage,
});

function SettingsPage() {
  const [url, setUrl] = useState("");
  const [mocks, setMocks] = useState(true);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    setUrl(getApiBaseUrl());
    setMocks(getUseMocks());
  }, []);

  async function test() {
    setTesting(true);
    setHealth(null);
    try {
      const h = await api.health();
      setHealth(h);
    } finally {
      setTesting(false);
    }
  }

  function save() {
    setApiBaseUrl(url.trim());
    setUseMocks(mocks);
    test();
  }

  return (
    <div className="px-6 py-6 max-w-[820px] mx-auto space-y-5">
      <div>
        <h1 className="text-[20px] font-semibold tracking-tight">Settings</h1>
        <p className="text-[13px] text-muted-foreground mt-0.5">
          Connection and technical configuration.
        </p>
      </div>

      <Section title="API Connection" subtitle="Base URL of the Python KGQA backend.">
        <label className="block text-[12px] font-medium mb-1">API base URL</label>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://kgqa.example.com"
          className="w-full rounded-md border border-input bg-card px-3 py-2 text-[13px] font-mono outline-none focus:border-ring focus:ring-2 focus:ring-ring/20"
        />
        <label className="mt-3 flex items-center gap-2 text-[12px]">
          <input type="checkbox" checked={mocks} onChange={(e) => setMocks(e.target.checked)} />
          Use mock responses (no API required)
        </label>
        <div className="mt-3 flex items-center gap-2">
          <button
            onClick={save}
            className="rounded-md bg-primary text-primary-foreground text-[12px] font-medium px-3 py-1.5 hover:opacity-90"
          >
            Save & test
          </button>
          <button
            onClick={test}
            disabled={testing}
            className="rounded-md border border-input text-[12px] font-medium px-3 py-1.5 hover:bg-accent disabled:opacity-50 inline-flex items-center gap-1.5"
          >
            {testing && <Loader2 className="h-3 w-3 animate-spin" />} Test connection
          </button>
        </div>
      </Section>

      <Section
        title="Backend Status"
        subtitle="Reported by the Python API. Fuseki is never accessed directly from the browser."
      >
        {!health ? (
          <div className="text-[12px] text-muted-foreground">Run a test to see status.</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
            <StatusRow label="API" status={health.api} />
            <StatusRow label="Fuseki (via API)" status={health.fuseki} />
            <StatusRow label="LLM" status={health.llm} />
            <div className="md:col-span-3 grid grid-cols-2 gap-3 text-[12px] mt-1">
              <div>
                <div className="text-muted-foreground text-[11px]">Version</div>
                <div className="font-mono">{health.version}</div>
              </div>
              <div>
                <div className="text-muted-foreground text-[11px]">Latency</div>
                <div className="font-mono">{health.latencyMs} ms</div>
              </div>
            </div>
          </div>
        )}
      </Section>

      <Section title="Endpoints" subtitle="Wired against the placeholder REST contract.">
        <ul className="space-y-1 text-[12px] font-mono">
          {[
            "POST   /api/questions",
            "GET    /api/autocomplete?q=…",
            "GET    /api/examples",
            "GET    /api/capabilities",
            "POST   /api/clarifications/{case_id}",
            "GET    /api/graph/ontology",
            "GET    /api/graph/data",
            "GET    /api/graph/evidence/{case_id}",
            "GET    /api/metrics/confidence",
            "GET    /api/health",
          ].map((line) => (
            <li key={line} className="text-muted-foreground">
              {line}
            </li>
          ))}
        </ul>
      </Section>
    </div>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-border bg-card">
      <div className="px-4 py-3 border-b border-border">
        <div className="text-[13px] font-semibold">{title}</div>
        {subtitle && <div className="text-[11px] text-muted-foreground mt-0.5">{subtitle}</div>}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function StatusRow({ label, status }: { label: string; status: "ok" | "degraded" | "down" }) {
  const Icon = status === "ok" ? CheckCircle2 : status === "degraded" ? AlertTriangle : XCircle;
  const color =
    status === "ok"
      ? "var(--success)"
      : status === "degraded"
        ? "var(--warning-foreground)"
        : "var(--destructive)";
  return (
    <div className="flex items-center gap-2 rounded border border-border p-2.5">
      <Icon className="h-4 w-4" style={{ color }} />
      <div>
        <div className="text-[12px] font-medium">{label}</div>
        <div className="text-[10px] uppercase tracking-wide" style={{ color }}>
          {status}
        </div>
      </div>
    </div>
  );
}
