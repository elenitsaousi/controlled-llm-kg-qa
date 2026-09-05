import type {
  Capability,
  ConfidenceMetrics,
  Example,
  GraphPayload,
  HealthStatus,
  QuestionResponse,
  Suggestion,
} from "./types";
import {
  mockCapabilities,
  mockEvidenceFor,
  mockExamples,
  mockHealth,
  mockMetrics,
  mockOntology,
  mockSuggestions,
  pickMockAnswer,
  resolveClarification,
} from "./mocks";

const LS_BASE_URL_KEY = "tdkgqa.apiBaseUrl";
const LS_USE_MOCKS_KEY = "tdkgqa.useMocks";

export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
  if (typeof window === "undefined") return configured;
  return window.localStorage.getItem(LS_BASE_URL_KEY) ?? configured;
}

export function setApiBaseUrl(url: string) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LS_BASE_URL_KEY, url);
}

export function getUseMocks(): boolean {
  if (typeof window === "undefined") return false;
  const v = window.localStorage.getItem(LS_USE_MOCKS_KEY);
  return v === "true";
}

export function setUseMocks(value: boolean) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LS_USE_MOCKS_KEY, String(value));
}

async function delay<T>(value: T, ms = 200): Promise<T> {
  return new Promise((res) => setTimeout(() => res(value), ms));
}

async function request<T>(path: string, init?: RequestInit, fallback?: T): Promise<T> {
  const base = getApiBaseUrl();
  if (getUseMocks() || !base) {
    if (fallback === undefined) throw new Error("No mock fallback available for " + path);
    return delay(fallback);
  }
  const res = await fetch(`${base}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`API error ${res.status} on ${path}`);
  return (await res.json()) as T;
}

export const api = {
  async ask(question: string): Promise<QuestionResponse> {
    const mock = {
      ...pickMockAnswer(question),
      responseTimeMs: 300 + Math.round(Math.random() * 600),
    };
    return request<QuestionResponse>(
      `/api/questions`,
      { method: "POST", body: JSON.stringify({ question }) },
      mock,
    );
  },
  async autocomplete(q: string, context = ""): Promise<Suggestion[]> {
    const lc = q.toLowerCase().trim();
    const filtered = lc
      ? mockSuggestions.filter((s) => s.label.toLowerCase().includes(lc))
      : mockSuggestions.slice(0, 6);
    return request<Suggestion[]>(
      `/api/autocomplete?q=${encodeURIComponent(q)}&context=${encodeURIComponent(context)}`,
      undefined,
      filtered,
    );
  },
  async examples(): Promise<Example[]> {
    return request<Example[]>(`/api/examples`, undefined, mockExamples);
  },
  async capabilities(): Promise<Capability[]> {
    return request<Capability[]>(`/api/capabilities`, undefined, mockCapabilities);
  },
  async clarify(caseId: string, choiceId: string): Promise<QuestionResponse> {
    return request<QuestionResponse>(
      `/api/clarifications/${caseId}`,
      { method: "POST", body: JSON.stringify({ choiceId }) },
      resolveClarification(choiceId),
    );
  },
  async ontology(): Promise<GraphPayload> {
    return request<GraphPayload>(`/api/graph/ontology`, undefined, mockOntology);
  },
  async dataGraph(): Promise<GraphPayload> {
    return request<GraphPayload>(`/api/graph/data?limit=500`, undefined, mockOntology);
  },
  async evidence(caseId: string): Promise<GraphPayload> {
    return request<GraphPayload>(
      `/api/graph/evidence/${caseId}`,
      undefined,
      mockEvidenceFor(caseId),
    );
  },
  async metrics(): Promise<ConfidenceMetrics> {
    return request<ConfidenceMetrics>(`/api/metrics/confidence`, undefined, mockMetrics);
  },
  async health(): Promise<HealthStatus> {
    return request<HealthStatus>(`/api/health`, undefined, mockHealth);
  },
};
