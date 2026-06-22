import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  MessageSquare,
  Network,
  Settings as SettingsIcon,
} from "lucide-react";
import { api } from "@/lib/api";
import type { HealthStatus } from "@/lib/types";

const NAV = [
  { to: "/ask", label: "Ask", icon: MessageSquare },
  { to: "/explorer", label: "Knowledge Graph", icon: Network },
  { to: "/confidence", label: "Confidence", icon: BarChart3 },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
] as const;

export function AppShell() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      api
        .health()
        .then((h) => !cancelled && setHealth(h))
        .catch(() => {});
    load();
    const id = setInterval(load, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const statusColor =
    health?.api === "ok"
      ? "var(--success)"
      : health?.api === "degraded"
        ? "var(--warning)"
        : health
          ? "var(--destructive)"
          : "var(--muted-foreground)";

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <aside className="w-60 shrink-0 border-r border-sidebar-border bg-sidebar flex flex-col">
        <div className="px-5 py-5 border-b border-sidebar-border">
          <div className="flex items-center gap-2">
            <div
              className="h-7 w-7 rounded-md flex items-center justify-center"
              style={{
                background:
                  "linear-gradient(135deg, var(--primary), color-mix(in oklab, var(--primary) 60%, var(--chart-2)))",
              }}
            >
              <Network className="h-4 w-4 text-primary-foreground" strokeWidth={2.5} />
            </div>
            <div className="leading-tight">
              <div className="text-[13px] font-semibold tracking-tight">True Demand</div>
              <div className="text-[11px] text-muted-foreground">KG QA</div>
            </div>
          </div>
        </div>
        <nav className="flex-1 px-2 py-3 space-y-0.5">
          {NAV.map(({ to, label, icon: Icon }) => {
            const active = pathname === to || (to !== "/ask" && pathname.startsWith(to));
            return (
              <Link
                key={to}
                to={to}
                className="flex items-center gap-2.5 px-3 py-2 rounded-md text-[13px] font-medium transition-colors"
                style={{
                  backgroundColor: active ? "var(--sidebar-accent)" : "transparent",
                  color: active ? "var(--sidebar-accent-foreground)" : "var(--sidebar-foreground)",
                }}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            );
          })}
        </nav>
        <div className="px-4 py-3 border-t border-sidebar-border text-[11px] text-muted-foreground">
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: statusColor }} />
            <span>{health ? `API ${health.api} · v${health.version}` : "Connecting…"}</span>
          </div>
          {health && (
            <div className="mt-1 flex items-center gap-3 pl-3.5">
              <span>Fuseki: {health.fuseki}</span>
              <span>LLM: {health.llm}</span>
            </div>
          )}
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        <header className="h-12 border-b bg-card/60 backdrop-blur sticky top-0 z-10 flex items-center px-6 gap-3">
          <Activity className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-[12px] text-muted-foreground">Enterprise Knowledge Graph QA</span>
        </header>
        <Outlet />
      </main>
    </div>
  );
}
