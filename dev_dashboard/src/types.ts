// Shared types for the dev dashboard. These mirror the /dev API contract.

export type ThinkingMode =
  | "none"
  | "disabled"
  | "adaptive"
  | "minimal"
  | "low"
  | "medium"
  | "High"
  | "high"
  | "xhigh"
  | "max";
export type RouterId = "gemini_native" | "claude_native" | "openrouter";
export type FlowType = "agent_loop" | "direct" | "multi_image" | "designer";
export type StepKind =
  | "prompt_initial"
  | "image_gen"
  | "eval"
  | "prompt_retry"
  | "direct_gen"
  | "designer_plan"
  | "designer_search"
  | "designer_sources"
  | "designer_get_image"
  | "designer_status"
  | "designer_input";
export type ProcessStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";
export type RunGroupStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "partial";

export interface CatalogImage {
  id: string | number;
  type: "design" | "product" | "area" | string;
  area?: string;
  products?: string[];
  thumb_url: string;
  full_url: string;
  name: string;
  filename?: string;
}

export interface CatalogArea {
  id: string;
  name: string;
  [k: string]: unknown;
}

export interface CatalogProduct {
  id: string;
  name: string;
  [k: string]: unknown;
}

export interface ModelInfo {
  id: string;
  router?: RouterId;
  pair_id?: string;
  model_id?: string;
  roles: string[];
  thinking_modes: ThinkingMode[];
}

export interface RouterInfo {
  id: RouterId;
  label: string;
  env_key?: string;
  supports_images?: boolean;
}

export interface ModelRouterPair {
  id: string;
  router: RouterId;
  model_id: string;
  label: string;
  roles: string[];
  thinking_modes: ThinkingMode[];
  source?: string;
  supports_reasoning?: boolean;
}

export interface UploadedImage {
  id: string;
  url: string;
  name: string;
}

export interface TestCaseImage {
  url: string;
  role: "user" | "ducon";
  metadata?: Record<string, unknown>;
}

export interface TestCase {
  id: string;
  name: string;
  user_images: TestCaseImage[];
  ducon_images: TestCaseImage[];
  use_ducon_data: boolean;
  hint?: string;
  created_at?: string;
}

export interface Combo {
  id: string;
  name: string;
  flow: FlowType;
  image_model_pair?: ModelRouterPair;
  image_model: string;
  image_thinking: ThinkingMode;
  prompt_model_pair?: ModelRouterPair;
  prompt_model: string;
  prompt_thinking: ThinkingMode;
  max_eval_rounds: number;
  max_prompt_verify_rounds: number;
  system_prompt_override?: string;
  aspect_ratio?: string;
  created_at?: string;
  designer_tool_access?: string[];
  designer_system_prompt_mode?: "append" | "compose" | "replace";
  designer_filesystem_root?: string;
  designer_max_generation_rounds?: number;
  designer_max_turns?: number;
  designer_wall_clock_budget_s?: number;
}

export interface DesignerJobFinal {
  job_id: string;
  design_generation?: {
    output_images?: { index: number; url: string }[];
    model_used?: string;
    approved?: boolean;
  };
  design_plan?: Record<string, unknown>;
  sources_used?: Record<string, unknown>[];
  elements_used?: unknown[];
  metadata?: Record<string, unknown>;
}

export interface DesignerJobEvent {
  type: string;
  job_id: string;
  status: string;
  created_at?: number;
  message?: string;
  plan?: Record<string, unknown>;
  sources?: Record<string, unknown>[];
  removed?: Record<string, unknown>[];
  output_images?: { index: number; url: string }[];
  final?: DesignerJobFinal;
  // Newer multi-round designer events (added by the backend in parallel).
  // The interface stays permissive via [k: string]: unknown; these are explicit
  // for clarity and so buildMessages can read them without casts.
  round?: number;
  approved?: boolean;
  reasons?: string[];
  defects?: string[];
  url?: string;
  name?: string;
  prompt?: string;
  // Agentic-loop events (assistant_message / tool_call / tool_result).
  turn?: number;
  text?: string;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface DesignerJobSnapshot {
  id: string;
  status: string;
  created_at?: number;
  input?: Record<string, unknown>;
  events: DesignerJobEvent[];
  final?: DesignerJobFinal;
  error?: string;
}

export interface SystemPromptSection {
  id: string;
  label: string;
  path: string | null;
  content: string;
}

export interface DesignerConfig {
  tools: string[];
  default_tool_access?: string[];
  tool_descriptions: Record<string, string>;
  filesystem: {
    implementation: string;
    operations: string[];
    notes: string[];
  };
  generation?: {
    default_max_eval_rounds?: number;
    default_max_prompt_verify_rounds?: number;
    default_max_generation_rounds?: number;
    default_aspect_ratio?: string;
    aspect_ratio_options?: string[];
  };
  session?: Record<string, unknown> & {
    context_policies?: string[];
    context_policy_meta?: SessionConfigSchema["context_policy_meta"];
    summarizer_modes?: string[];
    claude_compaction_modes?: string[];
    default_summarizer_instructions?: string;
    limits?: SessionConfigSchema["limits"];
    schema?: SessionConfigSchema;
    notes?: string[];
  };
  system_prompt: {
    sections: SystemPromptSection[];
    composed_default: string;
    max_context_chars: number;
  };
}

export interface SessionConfigFieldSchema {
  key: string;
  label: string;
  description: string;
  type: "integer" | "number" | "enum" | "boolean" | "text";
  min?: number;
  max?: number;
  step?: number;
  default?: number | string | boolean;
  options?: { value: string; label: string; description?: string }[];
  routers?: RouterId[];
  hide_when?: Record<string, string[]>;
  show_when?: Record<string, string[]>;
  scope?: string;
}

export interface SessionConfigSchema {
  routers: RouterId[];
  limits: Record<
    string,
    { min?: number; max?: number; default?: number; step?: number }
  >;
  context_policies: string[];
  context_policy_meta: Record<
    string,
    { label: string; description: string; routers?: RouterId[] }
  >;
  summarizer_modes: string[];
  claude_compaction_modes: string[];
  default_summarizer_instructions: string;
  fields: SessionConfigFieldSchema[];
  applicability: Record<string, RouterId[]>;
  notes: string[];
}

export interface Step {
  index: number;
  kind: StepKind;
  model?: string;
  thinking?: ThinkingMode;
  status: "running" | "completed" | "failed" | "skipped";
  started_at?: string;
  ended_at?: string;
  duration_ms?: number;
  prompt_used?: string;
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number;
  error?: string;
}

export interface ProcessResult {
  output_images: { url: string }[];
  total_duration_ms?: number;
  total_cost_usd?: number;
  cost_breakdown?: { model: string; cost_usd: number; tokens: number }[];
  retries?: number;
  final_prompt?: string;
  input_images?: { url: string; role: string; metadata?: Record<string, unknown> }[];
}

export interface Process {
  id: string;
  test_case_id: string;
  combo_id: string;
  status: ProcessStatus;
  steps: Step[];
  result?: ProcessResult;
  started_at?: string;
  ended_at?: string;
}

export interface RunGroupSnapshot {
  id: string;
  status: RunGroupStatus;
  processes: Process[];
  created_at?: string;
}

export interface RunGroupHistoryEntry {
  id: string;
  kind?: "benchmark";
  created_at: string;
  test_case_count: number;
  combo_count: number;
  status: RunGroupStatus;
}

export interface DesignerHistoryEntry {
  id: string;
  kind: "designer";
  created_at: number | string;
  status: ProcessStatus | string;
  prompt?: string;
  upload_url?: string;
  upload_name?: string;
  error?: string;
}

export interface DesignerSessionState {
  prompt: string;
  systemPrompt: string;
  promptMode: "append" | "compose" | "replace";
  sectionOverrides: Record<string, string>;
  filesystemRoot: string;
  aspectRatio: string;
  modelPairId: string;
  imagePairId: string;
  thinking: ThinkingMode;
  imageThinking: ThinkingMode;
  toolAccess: string[];
  maxGenerations?: number;
  maxTurns?: number;
  unlimitedTurns?: boolean;
  wallClockBudgetMinutes?: number;
  unlimitedWallClock?: boolean;
  /** Session / context management (see GET /designer/config session defaults). */
  maxTokens?: number;
  evalMaxTokens?: number;
  maxMessages?: number;
  maxToolResultChars?: number;
  retainRecentImageTurns?: number;
  contextTokenBudget?: number;
  contextTriggerRatio?: number;
  contextPolicy?: string;
  openrouterContextCompression?: boolean;
  claudeCompaction?: string;
  claudeCompactionTriggerTokens?: number;
  claudeCompactionInstructions?: string;
  summarizerMode?: string;
  summarizerInstructions?: string;
  maxEvalRounds?: number;
  maxPromptVerifyRounds?: number;
  uploadUrl?: string;
  uploadName?: string;
  jobId: string | null;
  events: DesignerJobEvent[];
  final: DesignerJobFinal | null;
  running: boolean;
  error: string | null;
  streamConnected: boolean;
}

// SSE event payload (one `data:` line per event)
export interface StreamEvent {
  type:
    | "snapshot"
    | "process_started"
    | "step_started"
    | "step_completed"
    | "process_completed"
    | "process_failed"
    | "run_completed";
  run_group_id?: string;
  process_id?: string;
  step?: Partial<Step> & { index: number };
  result?: ProcessResult;
  snapshot?: RunGroupSnapshot;
  status?: ProcessStatus | RunGroupStatus;
  error?: string;
}
