import type { DesignerSessionState, RouterId, SessionConfigFieldSchema, SessionConfigSchema } from "../types";
import {
  clampSessionNumber,
  isSessionFieldKeyVisible,
  visibleContextPolicies,
} from "../sessionConfigUtils";

type SessionPatch = Partial<DesignerSessionState>;

const FIELD_PATCH: Record<string, keyof DesignerSessionState> = {
  max_tokens: "maxTokens",
  eval_max_tokens: "evalMaxTokens",
  max_messages: "maxMessages",
  max_tool_result_chars: "maxToolResultChars",
  retain_recent_image_turns: "retainRecentImageTurns",
  context_token_budget: "contextTokenBudget",
  context_trigger_ratio: "contextTriggerRatio",
  context_policy: "contextPolicy",
  openrouter_context_compression: "openrouterContextCompression",
  claude_compaction: "claudeCompaction",
  claude_compaction_trigger_tokens: "claudeCompactionTriggerTokens",
  claude_compaction_instructions: "claudeCompactionInstructions",
  summarizer_mode: "summarizerMode",
  summarizer_instructions: "summarizerInstructions",
  max_eval_rounds: "maxEvalRounds",
  max_prompt_verify_rounds: "maxPromptVerifyRounds",
};

function sessionGet(session: DesignerSessionState, key: string): unknown {
  const patchKey = FIELD_PATCH[key];
  if (!patchKey) return undefined;
  return session[patchKey as keyof DesignerSessionState];
}

function FieldHelp({ text }: { text: string }) {
  return (
    <p className="field-help" title={text}>
      {text}
    </p>
  );
}

function ConfigLabel({ label, description }: { label: string; description?: string }) {
  return (
    <label>
      {label}
      {description ? (
        <span className="info-tip" title={description} aria-label={description}>
          ⓘ
        </span>
      ) : null}
    </label>
  );
}

function UseMaxButton({
  max,
  onApply,
}: {
  max: number | undefined;
  onApply: (value: number) => void;
}) {
  if (max == null) return null;
  return (
    <button
      type="button"
      className="btn small"
      style={{ marginLeft: 8, padding: "2px 8px", fontSize: "0.75rem" }}
      title={`Set to maximum allowed value (${max})`}
      onClick={() => onApply(max)}
    >
      Use max
    </button>
  );
}

export default function SessionConfigPanel({
  schema,
  session,
  router,
  generationDefaults,
  onPatch,
}: {
  schema: SessionConfigSchema | undefined;
  session: DesignerSessionState;
  router: RouterId | undefined;
  generationDefaults?: { maxEval?: number; maxVerify?: number };
  onPatch: (patch: SessionPatch) => void;
}) {
  const values = {
    contextPolicy: session.contextPolicy,
    summarizerMode: session.summarizerMode,
    claudeCompaction: session.claudeCompaction,
    openrouterContextCompression: session.openrouterContextCompression,
  };

  const fieldByKey = new Map((schema?.fields ?? []).map((f) => [f.key, f]));
  const visible = (key: string) => isSessionFieldKeyVisible(key, router, values, schema);

  const policyMeta = visibleContextPolicies(schema, router);
  const policyOptions = Object.keys(policyMeta).length
    ? Object.keys(policyMeta)
    : schema?.context_policies ?? ["auto"];

  const renderInt = (field: SessionConfigFieldSchema, placeholder?: string) => {
    const patchKey = FIELD_PATCH[field.key];
    if (!patchKey) return null;
    const raw = sessionGet(session, field.key);
    const num = typeof raw === "number" ? raw : undefined;
    return (
      <div className="field" key={field.key}>
        <div className="spread" style={{ alignItems: "center" }}>
          <ConfigLabel label={field.label} description={field.description} />
          <UseMaxButton
            max={field.max}
            onApply={(max) => onPatch({ [patchKey]: max } as SessionPatch)}
          />
        </div>
        <FieldHelp text={field.description} />
        <input
          type="number"
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          value={num ?? ""}
          onChange={(e) => {
            const parsed = e.target.value ? Number(e.target.value) : undefined;
            onPatch({
              [patchKey]: clampSessionNumber(parsed, field.min, field.max),
            } as SessionPatch);
          }}
          placeholder={placeholder ?? String(field.default ?? "")}
        />
      </div>
    );
  };

  const limits = schema?.limits ?? {};

  return (
    <>
      {router && (
        <p className="small muted" style={{ marginTop: 8 }}>
          Showing session options for designer router{" "}
          <span className="mono">{router}</span>. Provider-only fields are hidden when not
          applicable.
        </p>
      )}
      <div className="grid two-col" style={{ marginTop: 12 }}>
        {visible("max_tokens") &&
          fieldByKey.get("max_tokens") &&
          renderInt(fieldByKey.get("max_tokens")!)}
        {visible("eval_max_tokens") &&
          fieldByKey.get("eval_max_tokens") &&
          renderInt(fieldByKey.get("eval_max_tokens")!)}

        {visible("context_policy") && (
          <div className="field">
            <ConfigLabel
              label={fieldByKey.get("context_policy")?.label ?? "Context policy"}
              description={fieldByKey.get("context_policy")?.description}
            />
            <FieldHelp
              text={
                fieldByKey.get("context_policy")?.description ??
                "How history is compacted before each model call."
              }
            />
            <select
              value={session.contextPolicy ?? "auto"}
              onChange={(e) => onPatch({ contextPolicy: e.target.value })}
            >
              {policyOptions.map((p) => (
                <option key={p} value={p} title={policyMeta[p]?.description}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        )}

        {visible("context_token_budget") &&
          fieldByKey.get("context_token_budget") &&
          renderInt(fieldByKey.get("context_token_budget")!)}
        {visible("context_trigger_ratio") &&
          fieldByKey.get("context_trigger_ratio") &&
          renderInt(fieldByKey.get("context_trigger_ratio")!)}
        {visible("max_messages") &&
          fieldByKey.get("max_messages") &&
          renderInt(fieldByKey.get("max_messages")!)}
        {visible("max_tool_result_chars") &&
          fieldByKey.get("max_tool_result_chars") &&
          renderInt(fieldByKey.get("max_tool_result_chars")!)}
        {visible("retain_recent_image_turns") &&
          fieldByKey.get("retain_recent_image_turns") &&
          renderInt(fieldByKey.get("retain_recent_image_turns")!)}

        {visible("summarizer_mode") && (
          <div className="field">
            <ConfigLabel
              label={fieldByKey.get("summarizer_mode")?.label ?? "Summarizer mode"}
              description={fieldByKey.get("summarizer_mode")?.description}
            />
            <FieldHelp
              text={
                fieldByKey.get("summarizer_mode")?.description ??
                "LLM summary of middle history when over budget."
              }
            />
            <select
              value={session.summarizerMode ?? "default"}
              onChange={(e) => onPatch({ summarizerMode: e.target.value })}
            >
              {(schema?.summarizer_modes ?? ["default", "custom", "disabled"]).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        )}

        {visible("claude_compaction") && (
          <div className="field">
            <ConfigLabel
              label={fieldByKey.get("claude_compaction")?.label ?? "Claude compaction"}
              description={fieldByKey.get("claude_compaction")?.description}
            />
            <FieldHelp
              text={
                fieldByKey.get("claude_compaction")?.description ??
                "Claude native server-side compaction beta."
              }
            />
            <select
              value={session.claudeCompaction ?? "auto"}
              onChange={(e) => onPatch({ claudeCompaction: e.target.value })}
            >
              {(schema?.claude_compaction_modes ?? ["auto", "enabled", "disabled"]).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        )}

        {visible("claude_compaction_trigger_tokens") &&
          fieldByKey.get("claude_compaction_trigger_tokens") &&
          renderInt(fieldByKey.get("claude_compaction_trigger_tokens")!)}

        {visible("max_eval_rounds") &&
          fieldByKey.get("max_eval_rounds") &&
          renderInt(
            fieldByKey.get("max_eval_rounds")!,
            String(generationDefaults?.maxEval ?? limits.max_eval_rounds?.default ?? 3),
          )}
        {visible("max_prompt_verify_rounds") &&
          fieldByKey.get("max_prompt_verify_rounds") &&
          renderInt(
            fieldByKey.get("max_prompt_verify_rounds")!,
            String(generationDefaults?.maxVerify ?? limits.max_prompt_verify_rounds?.default ?? 2),
          )}
      </div>

      {visible("openrouter_context_compression") && (
        <label className="check-label" style={{ marginTop: 12 }}>
          <input
            type="checkbox"
            checked={session.openrouterContextCompression !== false}
            onChange={(e) => onPatch({ openrouterContextCompression: e.target.checked })}
          />
          {fieldByKey.get("openrouter_context_compression")?.label ??
            "OpenRouter context-compression plugin"}
        </label>
      )}
      {visible("openrouter_context_compression") && fieldByKey.get("openrouter_context_compression") && (
        <FieldHelp text={fieldByKey.get("openrouter_context_compression")!.description} />
      )}

      {visible("summarizer_instructions") && (
        <div className="field" style={{ marginTop: 12 }}>
          <ConfigLabel
            label={
              fieldByKey.get("summarizer_instructions")?.label ?? "Custom summarizer instructions"
            }
            description={fieldByKey.get("summarizer_instructions")?.description}
          />
          <FieldHelp
            text={
              fieldByKey.get("summarizer_instructions")?.description ??
              "Used when summarizer mode is custom."
            }
          />
          <textarea
            value={session.summarizerInstructions ?? ""}
            onChange={(e) => onPatch({ summarizerInstructions: e.target.value })}
            placeholder={
              schema?.default_summarizer_instructions ?? "Summarize prior agent work…"
            }
            rows={4}
          />
        </div>
      )}

      {visible("claude_compaction_instructions") && (
        <div className="field" style={{ marginTop: 12 }}>
          <ConfigLabel
            label={
              fieldByKey.get("claude_compaction_instructions")?.label ??
              "Claude compaction instructions (optional override)"
            }
            description={fieldByKey.get("claude_compaction_instructions")?.description}
          />
          <FieldHelp
            text={
              fieldByKey.get("claude_compaction_instructions")?.description ??
              "Leave empty for Anthropic default compaction prompt."
            }
          />
          <textarea
            value={session.claudeCompactionInstructions ?? ""}
            onChange={(e) => onPatch({ claudeCompactionInstructions: e.target.value })}
            placeholder="Leave empty for Anthropic default compaction prompt"
            rows={2}
          />
        </div>
      )}
    </>
  );
}
