import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { useStore } from "../store";
import DesignerChat from "../components/DesignerChat";
import SessionConfigPanel from "../components/SessionConfigPanel";
import { openLightbox } from "../components/Lightbox";
import { Banner, Empty } from "../components/ui";
import type {
  DesignerConfig,
  ModelRouterPair,
  SystemPromptSection,
  ThinkingMode,
  SessionConfigSchema,
} from "../types";
import { routerFromPairId } from "../sessionConfigUtils";

const DEFAULT_TEXT_PAIR = "gemini_native:gemini-3-flash-preview";
const DEFAULT_IMAGE_PAIR = "gemini_native:gemini-3-pro-image-preview";

export default function DesignerAgent() {
  const modelPairs = useStore((s) => s.modelPairs);
  const backendUp = useStore((s) => s.backendUp);
  const session = useStore((s) => s.designerSession);
  const patchDesignerSession = useStore((s) => s.patchDesignerSession);
  const startDesignerJob = useStore((s) => s.startDesignerJob);
  const cancelDesignerJob = useStore((s) => s.cancelDesignerJob);

  const [config, setConfig] = useState<DesignerConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [sections, setSections] = useState<SystemPromptSection[]>([]);

  useEffect(() => {
    if (!backendUp) return;
    api.designerConfig()
      .then((data) => {
        setConfig(data);
        if (!session.toolAccess.length) {
          patchDesignerSession({
            toolAccess: data.default_tool_access?.length ? data.default_tool_access : data.tools,
          });
        }
        setSections(
          data.system_prompt.sections.map((s) => ({
            ...s,
            content: session.sectionOverrides[s.id] ?? s.content,
          })),
        );
        if (session.promptMode === "replace" && !session.systemPrompt) {
          patchDesignerSession({ systemPrompt: data.system_prompt.composed_default });
        }
        setConfigError(null);
      })
      .catch((e) => setConfigError((e as Error).message));
  }, [backendUp]); // eslint-disable-line react-hooks/exhaustive-deps

  const textPairs = useMemo(
    () => modelPairs.filter((p) => p.roles.includes("designer") || p.roles.includes("prompt")),
    [modelPairs],
  );
  const imagePairs = useMemo(
    () => modelPairs.filter((p) => p.roles.includes("image")),
    [modelPairs],
  );
  const modelPair = findPair(textPairs, session.modelPairId);
  const imagePair = findPair(imagePairs, session.imagePairId);
  const validThinking = pickThinking(modelPair, session.thinking);
  const validImageThinking = pickThinking(imagePair, session.imageThinking);
  const sessionSchema = (config?.session?.schema ??
    config?.session) as SessionConfigSchema | undefined;
  const designerRouter = modelPair?.router ?? routerFromPairId(session.modelPairId);

  const composedPreview = useMemo(() => {
    if (!config) return "";
    if (session.promptMode === "replace") return session.systemPrompt;
    const body = sections.map((s) => `[${s.label}]\n${s.content}`).join("\n\n");
    if (session.systemPrompt.trim()) {
      return `${body}\n\n[Dev override]\n${session.systemPrompt}`;
    }
    return body;
  }, [config, session.promptMode, session.systemPrompt, sections]);

  const toggleTool = (tool: string) => {
    const toolAccess = session.toolAccess.includes(tool)
      ? session.toolAccess.filter((t) => t !== tool)
      : [...session.toolAccess, tool];
    patchDesignerSession({ toolAccess });
  };

  const updateSection = (id: string, content: string) => {
    setSections((prev) => prev.map((s) => (s.id === id ? { ...s, content } : s)));
    if (session.promptMode === "compose") {
      const original = config?.system_prompt.sections.find((s) => s.id === id);
      const sectionOverrides = { ...session.sectionOverrides };
      if (original && original.content !== content) {
        sectionOverrides[id] = content;
      } else {
        delete sectionOverrides[id];
      }
      patchDesignerSession({ sectionOverrides });
    }
  };

  const start = async () => {
    if (!file || !modelPair || !imagePair) return;
    patchDesignerSession({
      modelPairId: modelPair.id,
      imagePairId: imagePair.id,
      thinking: validThinking,
      imageThinking: validImageThinking,
    });
    await startDesignerJob(file);
  };

  const previewUrl = file ? URL.createObjectURL(file) : session.uploadUrl;

  return (
    <div>
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Designer Agent</h2>
        <div className="row">
          {session.jobId && <span className="mono small muted">job {session.jobId}</span>}
          {session.streamConnected && <span className="badge completed">● streaming</span>}
          {session.running && !session.streamConnected && (
            <span className="badge pending">○ reconnecting</span>
          )}
        </div>
      </div>

      {!backendUp && <Banner kind="err">Backend not connected.</Banner>}
      {configError && <Banner kind="err">{configError}</Banner>}
      {session.error && <Banner kind="err">{session.error}</Banner>}

      <div className="grid cols-2">
        <div className="panel">
          <div className="grid cols-2">
            <div className="field">
              <label>User image input</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              {previewUrl && (
                <img
                  className="chat-upload-preview"
                  src={previewUrl}
                  alt="upload preview"
                  onClick={() => openLightbox(previewUrl)}
                />
              )}
            </div>
            <div className="field">
              <label>Aspect ratio</label>
              <select
                value={session.aspectRatio}
                onChange={(e) => patchDesignerSession({ aspectRatio: e.target.value })}
              >
                <option value="auto">auto (match photo)</option>
                <option value="1:1">1:1</option>
                <option value="4:3">4:3</option>
                <option value="3:4">3:4</option>
                <option value="16:9">16:9</option>
                <option value="9:16">9:16</option>
              </select>
            </div>
            <div className="field">
              <label>Max generations (budget)</label>
              <input
                type="number"
                min={1}
                value={session.maxGenerations ?? ""}
                onChange={(e) =>
                  patchDesignerSession({
                    maxGenerations: e.target.value ? Number(e.target.value) : undefined,
                  })
                }
                placeholder="default 5"
              />
            </div>
            <div className="field">
              <label>Max agent turns (budget)</label>
              <label className="check-label" style={{ marginBottom: 8 }}>
                <input
                  type="checkbox"
                  checked={!!session.unlimitedTurns}
                  onChange={(e) =>
                    patchDesignerSession({ unlimitedTurns: e.target.checked })
                  }
                />
                No turn limit — agent stops when it calls finish or stops using tools
              </label>
              <input
                type="number"
                min={1}
                value={session.maxTurns ?? ""}
                disabled={!!session.unlimitedTurns}
                onChange={(e) =>
                  patchDesignerSession({
                    maxTurns: e.target.value ? Number(e.target.value) : undefined,
                  })
                }
                placeholder="default 24"
              />
            </div>
            <div className="field">
              <label>Wall clock (minutes)</label>
              <label className="check-label" style={{ marginBottom: 8 }}>
                <input
                  type="checkbox"
                  checked={!!session.unlimitedWallClock}
                  onChange={(e) =>
                    patchDesignerSession({ unlimitedWallClock: e.target.checked })
                  }
                />
                No time limit
              </label>
              <input
                type="number"
                min={1}
                value={session.wallClockBudgetMinutes ?? ""}
                disabled={!!session.unlimitedWallClock}
                onChange={(e) =>
                  patchDesignerSession({
                    wallClockBudgetMinutes: e.target.value ? Number(e.target.value) : undefined,
                  })
                }
                placeholder="default 15"
              />
            </div>

            <div className="field">
              <label>Designer model/router pair</label>
              <PairSelect
                value={modelPair?.id ?? DEFAULT_TEXT_PAIR}
                options={textPairs}
                onChange={(pair) =>
                  patchDesignerSession({
                    modelPairId: pair.id,
                    thinking: pair.thinking_modes[0],
                  })
                }
              />
            </div>
            <div className="field">
              <label>Thinking mode</label>
              <ThinkingSelect
                pair={modelPair}
                value={validThinking}
                onChange={(thinking) => patchDesignerSession({ thinking })}
              />
            </div>
            <div className="field">
              <label>Image generation model/router pair</label>
              <PairSelect
                value={imagePair?.id ?? DEFAULT_IMAGE_PAIR}
                options={imagePairs}
                onChange={(pair) =>
                  patchDesignerSession({
                    imagePairId: pair.id,
                    imageThinking: pair.thinking_modes[0],
                  })
                }
              />
            </div>
            <div className="field">
              <label>Image thinking mode</label>
              <ThinkingSelect
                pair={imagePair}
                value={validImageThinking}
                onChange={(imageThinking) => patchDesignerSession({ imageThinking })}
              />
            </div>

          <details className="field">
            <summary>Session &amp; context (tokens, compaction, summarizer)</summary>
            <p className="small muted" style={{ marginTop: 8 }}>
              Each job builds an in-memory message list (not a DB session). Every turn appends
              assistant + tool messages; images are re-sent on later turns until compacted.
            </p>
            {(sessionSchema?.notes ?? (config?.session as { notes?: string[] })?.notes)?.map(
              (note) => (
                <p className="small muted" key={note} style={{ marginTop: 4 }}>
                  {note}
                </p>
              ),
            )}
            <SessionConfigPanel
              schema={sessionSchema}
              session={session}
              router={designerRouter}
              generationDefaults={{
                maxEval: config?.generation?.default_max_eval_rounds,
                maxVerify: config?.generation?.default_max_prompt_verify_rounds,
              }}
              onPatch={patchDesignerSession}
            />
          </details>
          </div>

          <div className="field">
            <label>Optional text prompt</label>
            <textarea
              value={session.prompt}
              onChange={(e) => patchDesignerSession({ prompt: e.target.value })}
              placeholder="Design direction, constraints, desired products..."
            />
          </div>

          <div className="field">
            <label>System prompt mode</label>
            <select
              value={session.promptMode}
              onChange={(e) => {
                const promptMode = e.target.value as "append" | "compose" | "replace";
                const patch: Partial<typeof session> = { promptMode };
                if (promptMode === "replace" && !session.systemPrompt && config) {
                  patch.systemPrompt = config.system_prompt.composed_default;
                }
                patchDesignerSession(patch);
              }}
            >
              <option value="append">Append override to production prompts</option>
              <option value="compose">Edit production prompt sections</option>
              <option value="replace">Replace entire system prompt</option>
            </select>
          </div>

          {session.promptMode === "replace" ? (
            <div className="field">
              <label>Full system prompt</label>
              <textarea
                className="prompt-editor"
                value={session.systemPrompt}
                onChange={(e) => patchDesignerSession({ systemPrompt: e.target.value })}
              />
            </div>
          ) : (
            <>
              {sections.map((section) => (
                <div className="field prompt-section" key={section.id}>
                  <label>
                    {section.label}
                    {section.path && (
                      <span className="mono small muted"> · {section.path}</span>
                    )}
                  </label>
                  <textarea
                    className="prompt-editor"
                    value={section.content}
                    onChange={(e) => updateSection(section.id, e.target.value)}
                    readOnly={session.promptMode === "append"}
                  />
                </div>
              ))}
              <div className="field">
                <label>Dev override (optional)</label>
                <textarea
                  value={session.systemPrompt}
                  onChange={(e) => patchDesignerSession({ systemPrompt: e.target.value })}
                  placeholder="Extra dev-only instructions appended after production prompts..."
                />
              </div>
            </>
          )}

          <details className="field">
            <summary>
              Composed system prompt preview ({composedPreview.length.toLocaleString()} chars)
            </summary>
            <pre className="prompt-preview">{composedPreview || "Loading..."}</pre>
          </details>

          <div className="field">
            <label>Scoped filesystem directory</label>
            <input
              value={session.filesystemRoot}
              onChange={(e) => patchDesignerSession({ filesystemRoot: e.target.value })}
              placeholder="C:\Users\design\Documents\DuconAgentScratch"
            />
            {config && (
              <p className="small muted" style={{ marginTop: 8 }}>
                {config.filesystem.implementation}. Operations:{" "}
                {config.filesystem.operations.join(", ")}. {config.filesystem.notes.join(" ")}
              </p>
            )}
          </div>

          <div className="field">
            <label>Tool access</label>
            <p className="small muted" style={{ marginTop: 0, marginBottom: 8 }}>
              Only checked tools are sent to the model as API tool definitions. Mentions of other
              tool names inside the system prompt (e.g. from old production prompts) are not
              callable unless listed here.
            </p>
            <div className="row wrap">
              {(config?.tools ?? []).map((tool) => (
                <label key={tool} className="check-label" title={config?.tool_descriptions[tool]}>
                  <input
                    type="checkbox"
                    checked={session.toolAccess.includes(tool)}
                    onChange={() => toggleTool(tool)}
                  />
                  {tool}
                </label>
              ))}
            </div>
          </div>

          <div className="row" style={{ gap: 8 }}>
            <button className="primary" disabled={!file || session.running} onClick={start}>
              {session.running ? "Running..." : "Run Designer Agent"}
            </button>
            {session.running && session.jobId && (
              <button
                type="button"
                className="danger"
                onClick={() => cancelDesignerJob(session.jobId!)}
              >
                Stop
              </button>
            )}
          </div>
        </div>

        <div className="panel designer-chat-panel">
          <h3 style={{ marginTop: 0 }}>Chat</h3>
          {session.events.length === 0 && !session.running ? (
            <Empty>Start a run to see uploads, tool calls, and responses.</Empty>
          ) : (
            <DesignerChat
              uploadUrl={session.uploadUrl}
              uploadName={session.uploadName}
              prompt={session.prompt}
              events={session.events}
              final={session.final}
              running={session.running}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function PairSelect({
  value,
  options,
  onChange,
}: {
  value: string;
  options: ModelRouterPair[];
  onChange: (pair: ModelRouterPair) => void;
}) {
  const all = options.length ? options : [];
  return (
    <select
      value={all.some((p) => p.id === value) ? value : all[0]?.id ?? ""}
      onChange={(e) => {
        const pair = all.find((p) => p.id === e.target.value);
        if (pair) onChange(pair);
      }}
    >
      {all.map((pair) => (
        <option key={pair.id} value={pair.id}>
          {pair.model_id} · {pair.router}
        </option>
      ))}
    </select>
  );
}

function ThinkingSelect({
  pair,
  value,
  onChange,
}: {
  pair?: ModelRouterPair;
  value: ThinkingMode;
  onChange: (value: ThinkingMode) => void;
}) {
  const modes = pair?.thinking_modes ?? ["none"];
  return (
    <select
      value={modes.includes(value) ? value : modes[0]}
      onChange={(e) => onChange(e.target.value as ThinkingMode)}
    >
      {modes.map((mode) => (
        <option key={mode} value={mode}>
          {mode}
        </option>
      ))}
    </select>
  );
}

function findPair(pairs: ModelRouterPair[], id: string): ModelRouterPair | undefined {
  return pairs.find((pair) => pair.id === id) ?? pairs[0];
}

function pickThinking(pair: ModelRouterPair | undefined, mode: ThinkingMode): ThinkingMode {
  const modes = pair?.thinking_modes ?? ["none"];
  return modes.includes(mode) ? mode : modes[0];
}
