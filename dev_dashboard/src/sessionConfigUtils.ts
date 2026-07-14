import type { DesignerSessionState, RouterId, SessionConfigFieldSchema, SessionConfigSchema } from "./types";

type SessionValues = Pick<
  DesignerSessionState,
  | "contextPolicy"
  | "summarizerMode"
  | "claudeCompaction"
  | "openrouterContextCompression"
>;

const ALL_ROUTERS: RouterId[] = ["gemini_native", "claude_native", "openrouter"];

const CAMEL_TO_SNAKE: Record<string, string> = {
  maxTokens: "max_tokens",
  evalMaxTokens: "eval_max_tokens",
  maxMessages: "max_messages",
  maxToolResultChars: "max_tool_result_chars",
  retainRecentImageTurns: "retain_recent_image_turns",
  contextTokenBudget: "context_token_budget",
  contextTriggerRatio: "context_trigger_ratio",
  contextPolicy: "context_policy",
  openrouterContextCompression: "openrouter_context_compression",
  claudeCompaction: "claude_compaction",
  claudeCompactionTriggerTokens: "claude_compaction_trigger_tokens",
  claudeCompactionInstructions: "claude_compaction_instructions",
  summarizerMode: "summarizer_mode",
  summarizerInstructions: "summarizer_instructions",
  maxEvalRounds: "max_eval_rounds",
  maxPromptVerifyRounds: "max_prompt_verify_rounds",
};

export function sessionFieldKey(camelKey: string): string {
  return CAMEL_TO_SNAKE[camelKey] ?? camelKey;
}

/** Derive router id from a model pair id such as `claude_native:claude-sonnet-4-6`. */
export function routerFromPairId(pairId: string | undefined): RouterId | undefined {
  if (!pairId) return undefined;
  const prefix = pairId.split(":")[0] as RouterId;
  return ALL_ROUTERS.includes(prefix) ? prefix : undefined;
}

export function resolveFieldRouters(
  field: SessionConfigFieldSchema | undefined,
  key: string,
  applicability?: Record<string, RouterId[]>,
): RouterId[] | undefined {
  if (field?.routers?.length) return field.routers;
  const fromApp = applicability?.[key];
  return fromApp?.length ? fromApp : undefined;
}

function sessionValueForKey(values: SessionValues, snakeKey: string): string | undefined {
  switch (snakeKey) {
    case "context_policy":
      return values.contextPolicy ?? "auto";
    case "summarizer_mode":
      return values.summarizerMode ?? "default";
    case "claude_compaction":
      return values.claudeCompaction ?? "auto";
    default:
      return undefined;
  }
}

function matchesWhen(
  when: Record<string, string[]> | undefined,
  values: SessionValues,
): boolean {
  if (!when) return true;
  return Object.entries(when).every(([key, allowed]) => {
    const current = sessionValueForKey(values, key) ?? "";
    return allowed.includes(current);
  });
}

function routerAllowsField(routers: RouterId[] | undefined, router: RouterId | undefined): boolean {
  if (!routers?.length) return true;
  if (!router) {
    // Provider-exclusive fields stay hidden until the designer router is known.
    return routers.length >= ALL_ROUTERS.length;
  }
  return routers.includes(router);
}

export function isSessionFieldVisible(
  field: SessionConfigFieldSchema,
  router: RouterId | undefined,
  values: SessionValues,
  applicability?: Record<string, RouterId[]>,
): boolean {
  const routers = resolveFieldRouters(field, field.key, applicability);
  if (!routerAllowsField(routers, router)) return false;
  if (field.hide_when && matchesWhen(field.hide_when, values)) return false;
  if (field.show_when && !matchesWhen(field.show_when, values)) return false;
  return true;
}

export function isSessionFieldKeyVisible(
  key: string,
  router: RouterId | undefined,
  values: SessionValues,
  schema: SessionConfigSchema | undefined,
): boolean {
  const field = schema?.fields?.find((f) => f.key === key);
  if (!field) {
    const routers = resolveFieldRouters(undefined, key, schema?.applicability);
    return routerAllowsField(routers, router);
  }
  return isSessionFieldVisible(field, router, values, schema?.applicability);
}

export function visibleContextPolicies(
  schema: SessionConfigSchema | undefined,
  router: RouterId | undefined,
): SessionConfigSchema["context_policy_meta"] {
  const meta = schema?.context_policy_meta ?? {};
  if (!router) return meta;
  const out: SessionConfigSchema["context_policy_meta"] = {};
  for (const [key, entry] of Object.entries(meta)) {
    if (!entry.routers?.length || entry.routers.includes(router)) {
      out[key] = entry;
    }
  }
  return out;
}

export function clampSessionNumber(
  value: number | undefined,
  min: number | undefined,
  max: number | undefined,
): number | undefined {
  if (value == null || Number.isNaN(value)) return undefined;
  let n = value;
  if (min != null) n = Math.max(min, n);
  if (max != null) n = Math.min(max, n);
  return n;
}
