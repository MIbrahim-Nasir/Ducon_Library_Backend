// Single API client for the /dev backend endpoints.
// All calls are typed and degrade gracefully when the backend is down.
import type {
  CatalogArea,
  CatalogImage,
  CatalogProduct,
  Combo,
  DesignerJobEvent,
  DesignerJobSnapshot,
  ModelInfo,
  ModelRouterPair,
  RouterInfo,
  RunGroupHistoryEntry,
  RunGroupSnapshot,
  TestCase,
  UploadedImage,
} from "./types";

const BASE = "/dev";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function req<T>(
  path: string,
  init?: RequestInit,
  expectJson = true,
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch (e) {
    throw new ApiError(
      `Network error (backend not connected): ${(e as Error).message}`,
      0,
    );
  }
  if (!res.ok) {
    let body = "";
    try {
      body = await res.text();
    } catch {
      /* ignore */
    }
    throw new ApiError(
      `HTTP ${res.status} ${res.statusText}${body ? ` — ${body.slice(0, 200)}` : ""}`,
      res.status,
    );
  }
  if (!expectJson) return undefined as unknown as T;
  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("application/json")) {
    // Backend returned a non-JSON body (e.g. HTML error page) — treat as failure.
    throw new ApiError(`Unexpected content-type: ${ct}`, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  // ---- Catalog ----
  catalogImages: () =>
    req<{ images: CatalogImage[] }>("/catalog/images"),
  catalogAreas: () => req<{ areas: CatalogArea[] }>("/catalog/areas"),
  catalogProducts: () =>
    req<{ products: CatalogProduct[] }>("/catalog/products"),

  // ---- Models ----
  models: () =>
    req<{
      models: ModelInfo[];
      model_pairs?: ModelRouterPair[];
      routers?: RouterInfo[];
      thinking_modes?: string[];
    }>("/models"),
  createModelPair: (pair: Pick<ModelRouterPair, "router" | "model_id"> & Partial<ModelRouterPair>) =>
    req<ModelRouterPair>("/model-pairs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pair),
    }),
  deleteModelPair: (id: string) =>
    req<void>(`/model-pairs/${encodeURIComponent(id)}`, { method: "DELETE" }, false),

  // ---- Upload ----
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<UploadedImage>("/upload", { method: "POST", body: fd });
  },

  // ---- Test cases ----
  createTestCase: (tc: Omit<TestCase, "id">) =>
    req<TestCase>("/test-cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tc),
    }),
  listTestCases: () => req<{ test_cases: TestCase[] }>("/test-cases"),
  deleteTestCase: (id: string) =>
    req<void>(`/test-cases/${id}`, { method: "DELETE" }, false),

  // ---- Combos ----
  createCombo: (c: Omit<Combo, "id">) =>
    req<Combo>("/combos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(c),
    }),
  listCombos: () => req<{ combos: Combo[] }>("/combos"),
  deleteCombo: (id: string) =>
    req<void>(`/combos/${id}`, { method: "DELETE" }, false),

  // ---- Runs ----
  startRun: (body: { test_case_ids: string[]; combo_ids: string[] }) =>
    req<{ run_group_id: string; processes: { id: string; test_case_id: string; combo_id: string }[] }>(
      "/runs",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    ),
  getRun: (id: string) => req<RunGroupSnapshot>(`/runs/${id}`),
  getRunResults: (id: string) =>
    req<RunGroupSnapshot>(`/runs/${id}/results`),

  // ---- History ----
  history: () =>
    req<{ run_groups: RunGroupHistoryEntry[]; designer_jobs: import("./types").DesignerHistoryEntry[] }>(
      "/history",
    ),
  getHistoryItem: (id: string) => req<RunGroupSnapshot>(`/history/${id}`),
  deleteRunGroup: (id: string) =>
    req<void>(`/history/${id}`, { method: "DELETE" }, false),

  // ---- Designer agent ----
  designerConfig: () => req<import("./types").DesignerConfig>("/designer/config"),
  sessionConfigSchema: () =>
    req<import("./types").SessionConfigSchema>("/session-config-schema"),
  startDesignerJob: (body: {
    file: File;
    prompt: string;
    model_pair_id: string;
    image_pair_id: string;
    thinking: string;
    image_thinking: string;
    system_prompt: string;
    system_prompt_mode: string;
    system_prompt_sections: Record<string, string>;
    tool_access: string[];
    filesystem_root: string;
    aspect_ratio: string;
    max_generation_rounds?: number;
    max_turns?: number;
    wall_clock_budget_s?: number;
    max_tokens?: number;
    eval_max_tokens?: number;
    max_messages?: number;
    max_tool_result_chars?: number;
    retain_recent_image_turns?: number;
    context_token_budget?: number;
    context_trigger_ratio?: number;
    context_policy?: string;
    openrouter_context_compression?: boolean | string;
    claude_compaction?: string;
    claude_compaction_trigger_tokens?: number;
    claude_compaction_instructions?: string;
    summarizer_mode?: string;
    summarizer_instructions?: string;
    max_eval_rounds?: number;
    max_prompt_verify_rounds?: number;
  }) => {
    const fd = new FormData();
    fd.append("file", body.file);
    fd.append("prompt", body.prompt);
    fd.append("model_pair_id", body.model_pair_id);
    fd.append("image_pair_id", body.image_pair_id);
    fd.append("thinking", body.thinking);
    fd.append("image_thinking", body.image_thinking);
    fd.append("system_prompt", body.system_prompt);
    fd.append("system_prompt_mode", body.system_prompt_mode);
    fd.append("system_prompt_sections", JSON.stringify(body.system_prompt_sections));
    fd.append("tool_access", JSON.stringify(body.tool_access));
    fd.append("filesystem_root", body.filesystem_root);
    fd.append("aspect_ratio", body.aspect_ratio);
    if (body.max_generation_rounds != null)
      fd.append("max_generation_rounds", String(body.max_generation_rounds));
    if (body.max_turns != null) fd.append("max_turns", String(body.max_turns));
    if (body.wall_clock_budget_s != null)
      fd.append("wall_clock_budget_s", String(body.wall_clock_budget_s));
    const appendOpt = (key: string, val: string | number | boolean | undefined) => {
      if (val === undefined || val === null || val === "") return;
      fd.append(key, typeof val === "boolean" ? (val ? "true" : "false") : String(val));
    };
    appendOpt("max_tokens", body.max_tokens);
    appendOpt("eval_max_tokens", body.eval_max_tokens);
    appendOpt("max_messages", body.max_messages);
    appendOpt("max_tool_result_chars", body.max_tool_result_chars);
    appendOpt("retain_recent_image_turns", body.retain_recent_image_turns);
    appendOpt("context_token_budget", body.context_token_budget);
    appendOpt("context_trigger_ratio", body.context_trigger_ratio);
    appendOpt("context_policy", body.context_policy);
    appendOpt("openrouter_context_compression", body.openrouter_context_compression);
    appendOpt("claude_compaction", body.claude_compaction);
    appendOpt("claude_compaction_trigger_tokens", body.claude_compaction_trigger_tokens);
    appendOpt("claude_compaction_instructions", body.claude_compaction_instructions);
    appendOpt("summarizer_mode", body.summarizer_mode);
    appendOpt("summarizer_instructions", body.summarizer_instructions);
    appendOpt("max_eval_rounds", body.max_eval_rounds);
    appendOpt("max_prompt_verify_rounds", body.max_prompt_verify_rounds);
    return req<{ job_id: string; status: string }>("/designer/jobs", {
      method: "POST",
      body: fd,
    });
  },
  getDesignerJob: (id: string) =>
    req<DesignerJobSnapshot>(`/designer/jobs/${id}`),
  listDesignerJobs: (activeOnly = false) =>
    req<{ designer_jobs: import("./types").DesignerHistoryEntry[] }>(
      `/designer/jobs${activeOnly ? "?active_only=true" : ""}`,
    ),
  deleteDesignerJob: (id: string) =>
    req<void>(`/designer/jobs/${id}`, { method: "DELETE" }, false),
  cancelDesignerJob: (id: string) =>
    req<{ cancelled: boolean; id: string }>(`/designer/jobs/${id}/cancel`, { method: "POST" }),

  // ---- Reveal (open image location in OS file manager) ----
  // Sends the same URL you'd use as an <img src> (e.g. "/dev/outputs/..." or
  // "/dev/uploads/...") to the backend, which resolves it to a disk path and
  // asks the OS to reveal it.
  reveal: (url: string) =>
    req<{ ok: boolean; path?: string; error?: string }>("/reveal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }),
};

// ---- SSE helper ----
// Opens an EventSource on /dev/runs/{id}/stream. Returns a close() function.
// `onEvent` is invoked for each parsed event payload. On terminal events or
// persistent errors, falls back to polling GET /dev/runs/{id}.
export function openRunStream(
  runGroupId: string,
  onEvent: (ev: import("./types").StreamEvent) => void,
  onState?: (readyState: number) => void,
  onPoll?: () => void,
): () => void {
  let es: EventSource | null = null;
  let closed = false;
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  let errorBackoff = 0;

  const schedulePoll = (delayMs = 2000) => {
    if (closed || !onPoll) return;
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = setTimeout(() => {
      pollTimer = null;
      if (!closed) onPoll();
    }, delayMs);
  };

  const open = () => {
    if (closed) return;
    es?.close();
    es = new EventSource(`${BASE}/runs/${runGroupId}/stream`);
    es.onmessage = (msg) => {
      errorBackoff = 0;
      try {
        const parsed = JSON.parse(msg.data) as import("./types").StreamEvent;
        onEvent(parsed);
        if (parsed.type === "run_completed") {
          closed = true;
          es?.close();
          onState?.(2);
        }
      } catch {
        /* ignore malformed lines */
      }
    };
    es.onopen = () => {
      errorBackoff = 0;
      onState?.(1);
    };
    es.onerror = () => {
      onState?.(es?.readyState ?? -1);
      if (closed) {
        es?.close();
        return;
      }
      // EventSource auto-reconnects; poll as fallback during gaps.
      errorBackoff = Math.min(errorBackoff + 1, 5);
      schedulePoll(1000 * errorBackoff);
    };
  };

  open();

  return () => {
    closed = true;
    if (pollTimer) clearTimeout(pollTimer);
    es?.close();
  };
}

export function openDesignerStream(
  jobId: string,
  onEvent: (ev: DesignerJobEvent) => void,
  onState?: (readyState: number) => void,
  onPoll?: () => void,
): () => void {
  let es: EventSource | null = null;
  let closed = false;
  let pollTimer: ReturnType<typeof setTimeout> | null = null;

  const schedulePoll = (delayMs = 2000) => {
    if (closed || !onPoll) return;
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = setTimeout(() => {
      pollTimer = null;
      if (!closed) onPoll();
    }, delayMs);
  };

  es = new EventSource(`${BASE}/designer/jobs/${jobId}/stream`);
  es.onmessage = (msg) => {
    try {
      const ev = JSON.parse(msg.data) as DesignerJobEvent;
      onEvent(ev);
      if (ev.type === "final" || ev.type === "error" || ev.type === "cancelled") {
        closed = true;
        es?.close();
        onState?.(2);
      }
    } catch {
      /* ignore */
    }
  };
  es.onopen = () => onState?.(1);
  es.onerror = () => {
    onState?.(es?.readyState ?? -1);
    schedulePoll();
  };

  return () => {
    closed = true;
    if (pollTimer) clearTimeout(pollTimer);
    es?.close();
  };
}
