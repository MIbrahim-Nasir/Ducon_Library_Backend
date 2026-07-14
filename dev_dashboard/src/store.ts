// Central app state (zustand). Keeps test cases + combos in localStorage as the
// source of truth for the UI, and merges with the backend when available.
import { create } from "zustand";
import { api, openRunStream, openDesignerStream, ApiError } from "./api";
import { normalizeTestCase, toBackendTestCase } from "./testCaseUtils";
import { dedupeCatalogImages } from "./catalogUtils";
import type {
  CatalogArea,
  CatalogImage,
  CatalogProduct,
  Combo,
  DesignerHistoryEntry,
  DesignerJobEvent,
  DesignerJobFinal,
  DesignerSessionState,
  ModelInfo,
  ModelRouterPair,
  Process,
  RouterInfo,
  RunGroupHistoryEntry,
  RunGroupSnapshot,
  StreamEvent,
  TestCase,
  ThinkingMode,
} from "./types";

const LS_TEST_CASES = "ducon_dev_test_cases";
const LS_COMBOS = "ducon_dev_combos";
const LS_ACTIVE_TAB = "ducon_dev_active_tab";
const LS_DESIGNER_SESSION = "ducon_dev_designer_session";

const DEFAULT_DESIGNER_SESSION: DesignerSessionState = {
  prompt: "",
  systemPrompt: "",
  promptMode: "append",
  sectionOverrides: {},
  filesystemRoot: "",
  aspectRatio: "auto",
  modelPairId: "gemini_native:gemini-3-flash-preview",
  imagePairId: "gemini_native:gemini-3-pro-image-preview",
  thinking: "medium",
  imageThinking: "medium",
  toolAccess: [],
  jobId: null,
  events: [],
  final: null,
  running: false,
  error: null,
  streamConnected: false,
};

/** Form/settings fields persisted in localStorage — not live job runtime state. */
const DESIGNER_SESSION_PERSIST_KEYS: (keyof DesignerSessionState)[] = [
  "prompt",
  "systemPrompt",
  "promptMode",
  "sectionOverrides",
  "filesystemRoot",
  "aspectRatio",
  "modelPairId",
  "imagePairId",
  "thinking",
  "imageThinking",
  "toolAccess",
  "maxGenerations",
  "maxTurns",
  "unlimitedTurns",
  "wallClockBudgetMinutes",
  "unlimitedWallClock",
  "maxTokens",
  "evalMaxTokens",
  "contextPolicy",
  "contextTokenBudget",
  "contextTriggerRatio",
  "maxMessages",
  "maxToolResultChars",
  "retainRecentImageTurns",
  "summarizerMode",
  "summarizerInstructions",
  "claudeCompaction",
  "claudeCompactionTriggerTokens",
  "claudeCompactionInstructions",
  "openrouterContextCompression",
  "maxEvalRounds",
  "maxPromptVerifyRounds",
];

function pickPersistedDesignerSession(
  session: DesignerSessionState,
): Partial<DesignerSessionState> {
  const out: Partial<DesignerSessionState> = {};
  for (const key of DESIGNER_SESSION_PERSIST_KEYS) {
    (out as Record<string, unknown>)[key] = session[key];
  }
  return out;
}

function loadDesignerSession(): DesignerSessionState {
  try {
    const raw = localStorage.getItem(LS_DESIGNER_SESSION);
    if (!raw) return { ...DEFAULT_DESIGNER_SESSION };
    const parsed = JSON.parse(raw) as Partial<DesignerSessionState>;
    // Only restore form/settings — each new run starts with empty chat/job state.
    return { ...DEFAULT_DESIGNER_SESSION, ...pickPersistedDesignerSession({ ...DEFAULT_DESIGNER_SESSION, ...parsed }) };
  } catch {
    return { ...DEFAULT_DESIGNER_SESSION };
  }
}

function saveDesignerSession(session: DesignerSessionState) {
  try {
    localStorage.setItem(LS_DESIGNER_SESSION, JSON.stringify(pickPersistedDesignerSession(session)));
  } catch {
    /* ignore */
  }
}

function loadActiveTab(): Tab {
  try {
    const raw = localStorage.getItem(LS_ACTIVE_TAB);
    if (raw && isTab(raw)) return raw;
  } catch {
    /* ignore */
  }
  return "test-cases";
}

function isTab(v: string): v is Tab {
  return [
    "test-cases",
    "combos",
    "designer-agent",
    "run-queue",
    "live",
    "results",
    "history",
  ].includes(v);
}

function loadLS<T>(key: string): T[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (key === LS_TEST_CASES && Array.isArray(parsed)) {
      return parsed.map((item) => normalizeTestCase(item)) as T[];
    }
    return parsed as T[];
  } catch {
    return [];
  }
}
function saveLS<T>(key: string, val: T[]) {
  try {
    localStorage.setItem(key, JSON.stringify(val));
  } catch {
    /* ignore quota errors */
  }
}

function uid(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

/** Drop stale running rows when a terminal step exists for the same index+kind. */
export function dedupeSteps(steps: import("./types").Step[]): import("./types").Step[] {
  const terminal = new Set<string>();
  for (const s of steps) {
    if (s.status === "completed" || s.status === "failed") {
      terminal.add(`${s.index}:${s.kind}`);
    }
  }
  return steps.filter(
    (s) =>
      s.status !== "running" || !terminal.has(`${s.index}:${s.kind}`),
  );
}

/**
 * Stable identity for a designer event, used to dedup SSE replay.
 * The backend's `emit` always sets `created_at` (time.time(), sub-second), so
 * `${type}:${created_at}:${job_id}` is stable across replays: the SAME event
 * re-yielded by `stream_job` produces the SAME key, so applyDesignerEvent can
 * replace it instead of appending a duplicate. Two genuinely distinct events
 * of the same type in one job cannot share a timestamp in practice (each emit
 * awaits async I/O), so this key does not collapse separate events.
 */
function designerEventKey(ev: DesignerJobEvent): string {
  const ts = ev.created_at;
  if (ts != null) return `${ev.type}:${ts}:${ev.job_id}`;
  // No timestamp (shouldn't happen with the current backend) — fall back to a
  // content fingerprint so the key is still stable across replays.
  return `${ev.type}:${ev.job_id}:${JSON.stringify(ev)}`;
}

/** Remove duplicate events from a list (later wins) and sort chronologically. */
function dedupDesignerEvents(events: DesignerJobEvent[]): DesignerJobEvent[] {
  const out: DesignerJobEvent[] = [];
  const indexByKey = new Map<string, number>();
  for (const ev of events) {
    const key = designerEventKey(ev);
    const pos = indexByKey.get(key);
    if (pos == null) {
      indexByKey.set(key, out.length);
      out.push(ev);
    } else {
      out[pos] = ev; // replace with the newer version
    }
  }
  out.sort((a, b) => (a.created_at ?? 0) - (b.created_at ?? 0));
  return out;
}

function upsertStep(
  steps: import("./types").Step[],
  step: Partial<import("./types").Step> & { index: number },
  status: import("./types").Step["status"],
): import("./types").Step[] {
  const idx = step.index;
  const kind = step.kind ?? "image_gen";
  const next = [...steps];
  const pos = next.findIndex((s) => s.index === idx && s.kind === kind);
  const prev =
    pos >= 0
      ? next[pos]
      : { index: idx, kind, status: "running" as const };
  const merged = {
    ...prev,
    ...step,
    index: idx,
    kind,
    status,
  };
  if (pos >= 0) next[pos] = merged;
  else next.push(merged);
  return dedupeSteps(next);
}

export type Tab =
  | "test-cases"
  | "combos"
  | "designer-agent"
  | "run-queue"
  | "live"
  | "results"
  | "history";

interface AppState {
  // ---- connection / metadata ----
  backendUp: boolean;
  backendError: string | null;
  lastChecked: number | null;

  models: ModelInfo[];
  modelPairs: ModelRouterPair[];
  routers: RouterInfo[];
  catalogImages: CatalogImage[];
  catalogAreas: CatalogArea[];
  catalogProducts: CatalogProduct[];

  // ---- entities ----
  testCases: TestCase[];
  combos: Combo[];

  // ---- run queue selection ----
  selectedTestCaseIds: Set<string>;
  selectedComboIds: Set<string>;

  // ---- current run ----
  runGroup: RunGroupSnapshot | null;
  processes: Record<string, Process>;
  streamConnected: boolean;

  // ---- history ----
  history: RunGroupHistoryEntry[];
  designerHistory: DesignerHistoryEntry[];

  // ---- designer agent session ----
  designerSession: DesignerSessionState;
  selectedDesignerJobId: string | null;
  activeDesignerJobs: Record<string, DesignerSessionState & { id: string; status: string; created_at?: number }>;

  // ---- navigation ----
  activeTab: Tab;
  selectedProcessId: string | null;

  // ---- actions ----
  ping: () => Promise<void>;
  refreshCatalog: () => Promise<void>;
  refreshModels: () => Promise<void>;
  saveModelPair: (pair: Pick<ModelRouterPair, "router" | "model_id"> & Partial<ModelRouterPair>) => Promise<ModelRouterPair | null>;

  refreshTestCases: () => Promise<void>;
  saveTestCase: (tc: Omit<TestCase, "id"> & { id?: string }) => Promise<void>;
  deleteTestCase: (id: string) => Promise<void>;

  refreshCombos: () => Promise<void>;
  saveCombo: (c: Omit<Combo, "id"> & { id?: string }) => Promise<void>;
  deleteCombo: (id: string) => Promise<void>;

  toggleTestCaseSelected: (id: string) => void;
  toggleComboSelected: (id: string) => void;
  selectAllTestCases: (on: boolean) => void;
  selectAllCombos: (on: boolean) => void;

  startRun: () => Promise<string | null>;
  loadRun: (id: string) => Promise<void>;
  closeStream: () => void;
  applyStreamEvent: (ev: StreamEvent) => void;
  refreshRunSnapshot: (id: string) => Promise<void>;

  refreshHistory: () => Promise<void>;
  deleteRunGroup: (id: string) => Promise<void>;

  patchDesignerSession: (patch: Partial<DesignerSessionState>) => void;
  startDesignerJob: (file: File) => Promise<string | null>;
  cancelDesignerJob: (id?: string) => Promise<void>;
  applyDesignerEvent: (ev: DesignerJobEvent) => void;
  setDesignerEvents: (events: DesignerJobEvent[]) => void;
  loadDesignerJob: (id: string) => Promise<void>;
  reconnectDesignerStream: (id?: string) => void;
  closeDesignerStream: () => void;
  deleteDesignerJob: (id: string) => Promise<void>;
  selectDesignerJob: (id: string | null) => void;

  setTab: (t: Tab) => void;
  selectProcess: (id: string | null) => void;
}

export const useStore = create<AppState>((set, get) => ({
  backendUp: false,
  backendError: null,
  lastChecked: null,

  models: [],
  modelPairs: [],
  routers: [],
  catalogImages: [],
  catalogAreas: [],
  catalogProducts: [],

  testCases: loadLS<TestCase>(LS_TEST_CASES),
  combos: loadLS<Combo>(LS_COMBOS),

  selectedTestCaseIds: new Set(),
  selectedComboIds: new Set(),

  runGroup: null,
  processes: {},
  streamConnected: false,

  history: [],
  designerHistory: [],

  designerSession: loadDesignerSession(),
  selectedDesignerJobId: null,
  activeDesignerJobs: {},

  activeTab: loadActiveTab(),
  selectedProcessId: null,

  // ---- connection ----
  ping: async () => {
    try {
      await api.models();
      set({ backendUp: true, backendError: null, lastChecked: Date.now() });
    } catch (e) {
      set({
        backendUp: false,
        backendError: e instanceof ApiError ? e.message : String(e),
        lastChecked: Date.now(),
      });
    }
  },

  refreshCatalog: async () => {
    try {
      const [imgs, areas, prods] = await Promise.all([
        api.catalogImages(),
        api.catalogAreas(),
        api.catalogProducts(),
      ]);
      set({
        catalogImages: dedupeCatalogImages(imgs.images),
        catalogAreas: areas.areas,
        catalogProducts: prods.products,
        backendUp: true,
        backendError: null,
      });
    } catch (e) {
      set({
        backendUp: false,
        backendError: e instanceof ApiError ? e.message : String(e),
      });
    }
  },

  refreshModels: async () => {
    try {
      const m = await api.models();
      const modelPairs =
        m.model_pairs ??
        m.models.map((model) => ({
          id: model.pair_id ?? `gemini_native:${model.id}`,
          router: model.router ?? "gemini_native",
          model_id: model.model_id ?? model.id,
          label: model.id,
          roles: model.roles,
          thinking_modes: model.thinking_modes,
        }));
      set({
        models: m.models,
        modelPairs,
        routers: m.routers ?? [],
        backendUp: true,
        backendError: null,
      });
    } catch (e) {
      set({
        backendUp: false,
        backendError: e instanceof ApiError ? e.message : String(e),
      });
    }
  },

  saveModelPair: async (pair) => {
    try {
      const saved = await api.createModelPair(pair);
      const modelPairs = get().modelPairs.filter((p) => p.id !== saved.id);
      modelPairs.push(saved);
      modelPairs.sort((a, b) => `${a.router}:${a.model_id}`.localeCompare(`${b.router}:${b.model_id}`));
      set({ modelPairs, backendUp: true, backendError: null });
      return saved;
    } catch (e) {
      set({
        backendUp: false,
        backendError: e instanceof ApiError ? e.message : String(e),
      });
      return null;
    }
  },

  // ---- test cases ----
  refreshTestCases: async () => {
    try {
      const r = await api.listTestCases();
      const remote = r.test_cases.map((t) =>
        normalizeTestCase(t as unknown as Record<string, unknown>, get().catalogImages),
      );
      // Merge: remote is authoritative where ids match; keep local-only ones too.
      const local = get().testCases;
      const byId = new Map<string, TestCase>();
      for (const t of local) byId.set(t.id, t);
      for (const t of remote) byId.set(t.id, t);
      const merged = Array.from(byId.values());
      set({ testCases: merged });
      saveLS(LS_TEST_CASES, merged);
    } catch {
      // Keep local copy as source of truth when backend is down.
    }
  },

  saveTestCase: async (tc) => {
    const id = tc.id ?? uid("tc");
    const full: TestCase = { ...tc, id } as TestCase;
    // Mirror to backend (best-effort) using backend inputs[] shape.
    try {
      const saved = await api.createTestCase(toBackendTestCase(full) as unknown as Omit<TestCase, "id">);
      Object.assign(full, normalizeTestCase(saved as unknown as Record<string, unknown>, get().catalogImages));
    } catch {
      /* keep local */
    }
    const list = get().testCases.filter((t) => t.id !== id);
    list.push(full);
    set({ testCases: list });
    saveLS(LS_TEST_CASES, list);
  },

  deleteTestCase: async (id) => {
    try {
      await api.deleteTestCase(id);
    } catch {
      /* keep going */
    }
    const list = get().testCases.filter((t) => t.id !== id);
    set({ testCases: list });
    saveLS(LS_TEST_CASES, list);
    const sel = new Set(get().selectedTestCaseIds);
    sel.delete(id);
    set({ selectedTestCaseIds: sel });
  },

  // ---- combos ----
  refreshCombos: async () => {
    try {
      const r = await api.listCombos();
      const remote = r.combos;
      const local = get().combos;
      const byId = new Map<string, Combo>();
      for (const c of local) byId.set(c.id, c);
      for (const c of remote) byId.set(c.id, c);
      const merged = Array.from(byId.values());
      set({ combos: merged });
      saveLS(LS_COMBOS, merged);
    } catch {
      /* keep local */
    }
  },

  saveCombo: async (c) => {
    const id = c.id ?? uid("combo");
    const full: Combo = { ...c, id } as Combo;
    try {
      const saved = await api.createCombo(full);
      Object.assign(full, saved);
    } catch {
      /* keep local */
    }
    const list = get().combos.filter((x) => x.id !== id);
    list.push(full);
    set({ combos: list });
    saveLS(LS_COMBOS, list);
  },

  deleteCombo: async (id) => {
    try {
      await api.deleteCombo(id);
    } catch {
      /* keep going */
    }
    const list = get().combos.filter((x) => x.id !== id);
    set({ combos: list });
    saveLS(LS_COMBOS, list);
    const sel = new Set(get().selectedComboIds);
    sel.delete(id);
    set({ selectedComboIds: sel });
  },

  // ---- selection ----
  toggleTestCaseSelected: (id) => {
    const sel = new Set(get().selectedTestCaseIds);
    if (sel.has(id)) sel.delete(id);
    else sel.add(id);
    set({ selectedTestCaseIds: sel });
  },
  toggleComboSelected: (id) => {
    const sel = new Set(get().selectedComboIds);
    if (sel.has(id)) sel.delete(id);
    else sel.add(id);
    set({ selectedComboIds: sel });
  },
  selectAllTestCases: (on) => {
    set({ selectedTestCaseIds: on ? new Set(get().testCases.map((t) => t.id)) : new Set() });
  },
  selectAllCombos: (on) => {
    set({ selectedComboIds: on ? new Set(get().combos.map((c) => c.id)) : new Set() });
  },

  // ---- run lifecycle ----
  startRun: async () => {
    const tcIds = Array.from(get().selectedTestCaseIds);
    const coIds = Array.from(get().selectedComboIds);
    if (!tcIds.length || !coIds.length) return null;
    let runGroupId: string;
    let seeds: { id: string; test_case_id: string; combo_id: string }[] = [];
    try {
      const r = await api.startRun({ test_case_ids: tcIds, combo_ids: coIds });
      runGroupId = r.run_group_id;
      seeds = r.processes;
    } catch {
      // Backend down — fabricate a local-only run group so the UI is still usable.
      runGroupId = uid("run");
      for (const tcId of tcIds) {
        for (const coId of coIds) {
          seeds.push({ id: uid("proc"), test_case_id: tcId, combo_id: coId });
        }
      }
    }
    const processes: Record<string, Process> = {};
    for (const s of seeds) {
      processes[s.id] = {
        id: s.id,
        test_case_id: s.test_case_id,
        combo_id: s.combo_id,
        status: "pending",
        steps: [],
      };
    }
    set({
      runGroup: {
        id: runGroupId,
        status: "running",
        processes: Object.values(processes),
      },
      processes,
      selectedProcessId: null,
      activeTab: "live",
    });
    // Kick off streaming + an immediate snapshot poll.
    get().loadRun(runGroupId);
    return runGroupId;
  },

  loadRun: async (id) => {
    // seed processes from current state if present, else fetch
    let processes = get().processes;
    if (Object.keys(processes).length === 0) {
      try {
        const snap = await api.getRun(id);
        processes = {};
        for (const p of snap.processes) {
          processes[p.id] = { ...p, steps: dedupeSteps(p.steps ?? []) };
        }
        set({
          runGroup: snap,
          processes,
          activeTab: "live",
        });
      } catch {
        set({ runGroup: { id, status: "running", processes: [] }, activeTab: "live" });
      }
    }
    // Open SSE stream (will no-op reconnect handled internally).
    closeStreamRef?.();
    closeStreamRef = openRunStream(
      id,
      (ev) => get().applyStreamEvent(ev),
      (rs) => set({ streamConnected: rs === 1 }),
      () => get().refreshRunSnapshot(id),
    );
  },

  closeStream: () => {
    closeStreamRef?.();
    closeStreamRef = null;
    set({ streamConnected: false });
  },

  applyStreamEvent: (ev) => {
    if (ev.type === "snapshot" && ev.snapshot) {
      const processes: Record<string, Process> = {};
      for (const p of ev.snapshot.processes) {
        processes[p.id] = { ...p, steps: dedupeSteps(p.steps ?? []) };
      }
      set({
        runGroup: ev.snapshot,
        processes,
        streamConnected: true,
      });
      return;
    }

    if (ev.type === "run_completed") {
      if (ev.run_group_id) {
        get().refreshRunSnapshot(ev.run_group_id);
      }
      set({ streamConnected: false });
      return;
    }

    if (!ev.process_id) return;

    const processes = { ...get().processes };
    const p = processes[ev.process_id];
    if (!p) return;
    const np: Process = { ...p };
    switch (ev.type) {
      case "process_started":
        np.status = "running";
        np.started_at = new Date().toISOString();
        break;
      case "step_started": {
        np.status = "running";
        if (ev.step) {
          np.steps = upsertStep(np.steps, ev.step, "running");
        }
        break;
      }
      case "step_completed": {
        if (ev.step) {
          np.steps = upsertStep(np.steps, ev.step, ev.step.status ?? "completed");
        }
        break;
      }
      case "process_completed":
        np.status = "completed";
        np.ended_at = new Date().toISOString();
        if (ev.result) np.result = ev.result;
        break;
      case "process_failed":
        np.status = "failed";
        np.ended_at = new Date().toISOString();
        if (ev.result) np.result = ev.result;
        if (ev.error) {
          const steps = [...np.steps];
          steps.push({
            index: steps.length,
            kind: "eval",
            status: "failed",
            error: ev.error,
            ended_at: new Date().toISOString(),
          });
          np.steps = steps;
        }
        break;
    }
    processes[ev.process_id] = np;
    set({ processes });
  },

  refreshRunSnapshot: async (id) => {
    try {
      const snap = await api.getRun(id);
      const processes: Record<string, Process> = {};
      for (const p of snap.processes) {
        processes[p.id] = { ...p, steps: dedupeSteps(p.steps ?? []) };
      }
      // Merge: keep local steps that are newer/running if backend lags.
      const cur = get().processes;
      for (const k of Object.keys(processes)) {
        const local = cur[k];
        if (local && dedupeSteps(local.steps).length > processes[k].steps.length) {
          processes[k] = { ...processes[k], steps: dedupeSteps(local.steps) };
        }
      }
      set({ runGroup: snap, processes });
    } catch {
      /* ignore */
    }
  },

  // ---- history ----
  refreshHistory: async () => {
    try {
      const r = await api.history();
      set({
        history: r.run_groups,
        designerHistory: r.designer_jobs ?? [],
        backendUp: true,
        backendError: null,
      });
    } catch (e) {
      set({
        backendUp: false,
        backendError: e instanceof ApiError ? e.message : String(e),
        history: [],
        designerHistory: [],
      });
    }
  },

  deleteRunGroup: async (id) => {
    try {
      await api.deleteRunGroup(id);
    } catch {
      /* keep going */
    }
    const history = get().history.filter((h) => h.id !== id);
    set({ history });
    if (get().runGroup?.id === id) {
      get().closeStream();
      set({ runGroup: null, processes: {}, selectedProcessId: null });
    }
  },

  patchDesignerSession: (patch) => {
    const designerSession = { ...get().designerSession, ...patch };
    set({ designerSession });
    saveDesignerSession(designerSession);
  },

  startDesignerJob: async (file) => {
    // Tear down any prior SSE stream so stale events cannot append to the new run.
    get().closeDesignerStream();
    const s = get().designerSession;
    const cleared: DesignerSessionState = {
      ...s,
      running: true,
      error: null,
      events: [],
      final: null,
      jobId: null,
      streamConnected: false,
      uploadUrl: undefined,
      uploadName: undefined,
    };
    set({ designerSession: cleared });
    saveDesignerSession(cleared);
    try {
      let uploadUrl = cleared.uploadUrl;
      let uploadName = cleared.uploadName;
      try {
        const uploaded = await api.upload(file);
        uploadUrl = uploaded.url;
        uploadName = uploaded.name;
      } catch {
        /* fall back to multipart in startDesignerJob */
      }
      const created = await api.startDesignerJob({
        file,
        prompt: s.prompt,
        model_pair_id: s.modelPairId,
        image_pair_id: s.imagePairId,
        thinking: s.thinking,
        image_thinking: s.imageThinking,
        system_prompt: s.systemPrompt,
        system_prompt_mode: s.promptMode,
        system_prompt_sections: s.promptMode === "compose" ? s.sectionOverrides : {},
        tool_access: s.toolAccess,
        filesystem_root: s.filesystemRoot,
        aspect_ratio: s.aspectRatio,
        max_generation_rounds: s.maxGenerations || undefined,
        max_turns: s.unlimitedTurns ? 0 : s.maxTurns || undefined,
        wall_clock_budget_s: s.unlimitedWallClock
          ? 0
          : s.wallClockBudgetMinutes != null
            ? s.wallClockBudgetMinutes * 60
            : undefined,
        max_tokens: s.maxTokens,
        eval_max_tokens: s.evalMaxTokens,
        max_messages: s.maxMessages,
        max_tool_result_chars: s.maxToolResultChars,
        retain_recent_image_turns: s.retainRecentImageTurns,
        context_token_budget: s.contextTokenBudget,
        context_trigger_ratio: s.contextTriggerRatio,
        context_policy: s.contextPolicy,
        openrouter_context_compression: s.openrouterContextCompression,
        claude_compaction: s.claudeCompaction,
        claude_compaction_trigger_tokens: s.claudeCompactionTriggerTokens,
        claude_compaction_instructions: s.claudeCompactionInstructions,
        summarizer_mode: s.summarizerMode,
        summarizer_instructions: s.summarizerInstructions,
        max_eval_rounds: s.maxEvalRounds,
        max_prompt_verify_rounds: s.maxPromptVerifyRounds,
      });
      const nextSession: DesignerSessionState = {
        ...get().designerSession,
        jobId: created.job_id,
        uploadUrl,
        uploadName,
        running: true,
        error: null,
        events: [],
        final: null,
      };
      set({ designerSession: nextSession });
      saveDesignerSession(nextSession);
      get().reconnectDesignerStream(created.job_id);
      get().refreshHistory();
      return created.job_id;
    } catch (e) {
      const err = e instanceof ApiError ? e.message : String(e);
      get().patchDesignerSession({ running: false, error: err });
      return null;
    }
  },

  applyDesignerEvent: (ev) => {
    const s = get().designerSession;
    // Ignore events from a superseded job (e.g. old SSE still open briefly).
    if (s.jobId && ev.job_id && ev.job_id !== s.jobId) return;
    // Dedup by stable identity key so SSE replay (stream_job yields ALL
    // accumulated events first, then live) and reconnects don't append the
    // same event twice. If the key already exists, replace it in place (the
    // backend may re-emit with more fields); otherwise append.
    const key = designerEventKey(ev);
    const events = [...s.events];
    const pos = events.findIndex((e) => designerEventKey(e) === key);
    if (pos >= 0) events[pos] = ev;
    else events.push(ev);
    events.sort((a, b) => (a.created_at ?? 0) - (b.created_at ?? 0));
    const patch: Partial<DesignerSessionState> = { events };
    if (ev.type === "final") {
      patch.final = ev as unknown as DesignerJobFinal;
      patch.running = false;
      patch.streamConnected = false;
    }
    if (ev.type === "error" || ev.type === "cancelled") {
      patch.error = String(ev.message || (ev.type === "cancelled" ? "Job cancelled." : "Designer job failed."));
      patch.running = false;
      patch.streamConnected = false;
    }
    const designerSession = { ...s, ...patch };
    set({ designerSession });
    saveDesignerSession(designerSession);
    if (designerSession.jobId) {
      const active = { ...get().activeDesignerJobs };
      active[designerSession.jobId] = {
        ...designerSession,
        id: designerSession.jobId,
        status: ev.status || (patch.running === false ? "completed" : "running"),
      };
      set({ activeDesignerJobs: active });
    }
    if (patch.running === false) {
      get().refreshHistory();
    }
  },

  setDesignerEvents: (events) => {
    // REPLACES the events array (deduped + sorted) — used by loadDesignerJob so
    // a server snapshot never accumulates on top of the local copy.
    const designerSession = {
      ...get().designerSession,
      events: dedupDesignerEvents(events),
    };
    set({ designerSession });
    saveDesignerSession(designerSession);
  },

  cancelDesignerJob: async (id) => {
    const jobId = id ?? get().designerSession.jobId;
    if (!jobId) return;
    try {
      await api.cancelDesignerJob(jobId);
    } catch (e) {
      const err = e instanceof ApiError ? e.message : String(e);
      get().patchDesignerSession({ error: err, running: false, streamConnected: false });
    }
    get().closeDesignerStream();
    if (get().designerSession.jobId === jobId) {
      get().patchDesignerSession({
        running: false,
        streamConnected: false,
        error: get().designerSession.error ?? "Job cancelled.",
      });
    }
    get().refreshHistory();
  },

  loadDesignerJob: async (id) => {
    try {
      get().closeDesignerStream();
      const snap = await api.getDesignerJob(id);
      const input = (snap as { input?: Record<string, unknown> }).input || {};
      // Replace events entirely (no accumulation). Snapshot events are deduped
      // and sorted by setDesignerEvents.
      get().setDesignerEvents(snap.events ?? []);
      const designerSession: DesignerSessionState = {
        ...get().designerSession,
        jobId: snap.id,
        final: snap.final ?? null,
        running: snap.status === "running" || snap.status === "queued",
        error: snap.error ?? null,
        prompt: String(input.prompt ?? get().designerSession.prompt),
        uploadUrl: String(input.upload_url ?? get().designerSession.uploadUrl ?? ""),
        uploadName: String(input.upload_name ?? get().designerSession.uploadName ?? ""),
      };
      set({ designerSession, selectedDesignerJobId: id });
      saveDesignerSession(designerSession);
      if (designerSession.running) {
        get().reconnectDesignerStream(id);
      }
    } catch {
      /* ignore */
    }
  },

  reconnectDesignerStream: (id) => {
    const jobId = id ?? get().designerSession.jobId;
    if (!jobId) return;
    // Guard: if we're already connected to this same job, don't reopen. This
    // prevents two simultaneous streams when multiple effects fire (e.g. on
    // reload) and avoids re-replaying the full event history.
    if (jobId === designerStreamJobId && closeDesignerStreamRef) return;
    closeDesignerStreamRef?.();
    closeDesignerStreamRef = openDesignerStream(
      jobId,
      (ev) => {
        // Ignore events from a superseded stream (defensive).
        if (designerStreamJobId !== jobId) return;
        get().applyDesignerEvent(ev);
      },
      (rs) => {
        if (designerStreamJobId !== jobId) return;
        get().patchDesignerSession({ streamConnected: rs === 1 });
      },
      () => {
        if (designerStreamJobId !== jobId) return;
        get().loadDesignerJob(jobId);
      },
    );
    designerStreamJobId = jobId;
  },

  closeDesignerStream: () => {
    closeDesignerStreamRef?.();
    closeDesignerStreamRef = null;
    designerStreamJobId = null;
    get().patchDesignerSession({ streamConnected: false });
  },

  deleteDesignerJob: async (id) => {
    try {
      await api.deleteDesignerJob(id);
    } catch {
      /* keep going */
    }
    const designerHistory = get().designerHistory.filter((j) => j.id !== id);
    const activeDesignerJobs = { ...get().activeDesignerJobs };
    delete activeDesignerJobs[id];
    set({ designerHistory, activeDesignerJobs });
    if (get().designerSession.jobId === id) {
      get().closeDesignerStream();
      get().patchDesignerSession({ jobId: null, running: false });
    }
    if (get().selectedDesignerJobId === id) {
      set({ selectedDesignerJobId: null });
    }
  },

  selectDesignerJob: (id) => {
    set({ selectedDesignerJobId: id, activeTab: id ? "designer-agent" : get().activeTab });
    if (id) get().loadDesignerJob(id);
  },

  setTab: (t) => {
    set({ activeTab: t });
    try {
      localStorage.setItem(LS_ACTIVE_TAB, t);
    } catch {
      /* ignore */
    }
  },
  selectProcess: (id) => set({ selectedProcessId: id }),
}));

// Hold the close fn outside the store to avoid serialization concerns.
let closeStreamRef: null | (() => void) = null;
let closeDesignerStreamRef: null | (() => void) = null;
// Tracks which job the current designer SSE stream is bound to, so concurrent
// reconnectDesignerStream calls for the same job don't open a second stream.
let designerStreamJobId: string | null = null;

// Helpers exposed for components.
export { uid };
