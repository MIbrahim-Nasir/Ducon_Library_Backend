import { useMemo, useState } from "react";
import { useStore } from "../store";
import { Banner, Empty, Modal } from "../components/ui";
import type { Combo, FlowType, ModelRouterPair, RouterId, ThinkingMode } from "../types";

const FLOWS: { id: FlowType; label: string }[] = [
  { id: "agent_loop", label: "agent_loop (prompt agent + Nano Banana + eval/retry)" },
  { id: "direct", label: "direct (fixed system prompt, no agent loop)" },
  { id: "multi_image", label: "multi_image (multi-image gen pipeline)" },
  { id: "designer", label: "designer (dev designer agent: plan + search + generate)" },
];

const DEFAULT_IMAGE_PAIR: ModelRouterPair = {
  id: "gemini_native:gemini-3-pro-image-preview",
  router: "gemini_native",
  model_id: "gemini-3-pro-image-preview",
  label: "gemini-3-pro-image-preview via Gemini native",
  roles: ["image", "prompt", "designer"],
  thinking_modes: ["minimal", "low", "medium", "High", "max"],
};

const DEFAULT_PROMPT_PAIR: ModelRouterPair = {
  id: "gemini_native:gemini-3-flash-preview",
  router: "gemini_native",
  model_id: "gemini-3-flash-preview",
  label: "gemini-3-flash-preview via Gemini native",
  roles: ["prompt", "designer"],
  thinking_modes: ["minimal", "low", "medium", "High", "max"],
};

const DEFAULT_COMBO: Omit<Combo, "id"> = {
  name: "",
  flow: "agent_loop",
  image_model_pair: DEFAULT_IMAGE_PAIR,
  image_model: DEFAULT_IMAGE_PAIR.model_id,
  image_thinking: "medium",
  prompt_model_pair: DEFAULT_PROMPT_PAIR,
  prompt_model: DEFAULT_PROMPT_PAIR.model_id,
  prompt_thinking: "medium",
  max_eval_rounds: 3,
  max_prompt_verify_rounds: 2,
  system_prompt_override: "",
  aspect_ratio: "auto",
  designer_tool_access: [
    "ai_search",
    "keyword_search",
    "get_image",
    "generate_multi_image",
    "filesystem",
  ],
  designer_system_prompt_mode: "append",
  designer_filesystem_root: "",
  designer_max_generation_rounds: 5,
  designer_max_turns: 24,
  designer_wall_clock_budget_s: 15 * 60,
};

export default function Combos() {
  const combos = useStore((s) => s.combos);
  const modelPairs = useStore((s) => s.modelPairs);
  const routers = useStore((s) => s.routers);
  const backendUp = useStore((s) => s.backendUp);
  const saveCombo = useStore((s) => s.saveCombo);
  const saveModelPair = useStore((s) => s.saveModelPair);
  const deleteCombo = useStore((s) => s.deleteCombo);
  const refreshCombos = useStore((s) => s.refreshCombos);
  const refreshModels = useStore((s) => s.refreshModels);

  const [editing, setEditing] = useState<Combo | null>(null);
  const [creating, setCreating] = useState(false);

  const pairs = useMemo(() => {
    return modelPairs.length ? modelPairs : [DEFAULT_IMAGE_PAIR, DEFAULT_PROMPT_PAIR];
  }, [modelPairs]);

  return (
    <div>
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Combinations</h2>
        <div className="row">
          <button onClick={() => refreshCombos()}>Refresh</button>
          <button
            className="primary"
            onClick={() => {
              setEditing(null);
              setCreating(true);
            }}
          >
            + New Combo
          </button>
        </div>
      </div>

      {!backendUp && (
        <Banner kind="err">
          Backend not connected — combos are read from local cache only.
        </Banner>
      )}

      <AddModelPair
        routers={routers.length ? routers.map((r) => r.id) : ["gemini_native", "claude_native", "openrouter"]}
        onSave={async (pair) => {
          await saveModelPair(pair);
          await refreshModels();
        }}
      />

      {combos.length === 0 ? (
        <Empty>No combos yet. Create one to define a model + thinking + flow configuration.</Empty>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Flow</th>
              <th>Image pair</th>
              <th>Prompt pair</th>
              <th>Rounds</th>
              <th>Aspect</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {combos.map((c) => (
              <tr key={c.id}>
                <td><strong>{c.name}</strong></td>
                <td><span className="tag-pill">{c.flow}</span></td>
                <td className="mono small">{c.image_model_pair?.id ?? c.image_model} <span className="muted">/{c.image_thinking}</span></td>
                <td className="mono small">{c.prompt_model_pair?.id ?? c.prompt_model} <span className="muted">/{c.prompt_thinking}</span></td>
                <td className="small">
                  eval {c.max_eval_rounds} · verify {c.max_prompt_verify_rounds}
                </td>
                <td className="small">{c.aspect_ratio ?? "—"}</td>
                <td>
                  <div className="row" style={{ justifyContent: "flex-end" }}>
                    <button
                      onClick={() => {
                        setEditing(c);
                        setCreating(true);
                      }}
                    >
                      Edit
                    </button>
                    <button
                      className="danger"
                      onClick={() => {
                        if (confirm(`Delete combo "${c.name}"?`)) deleteCombo(c.id);
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {creating && (
        <ComboEditor
          initial={editing}
          modelPairs={pairs}
          onClose={() => setCreating(false)}
          onSave={async (c) => {
            await saveCombo(c);
            setCreating(false);
          }}
        />
      )}
    </div>
  );
}

function ComboEditor({
  initial,
  modelPairs,
  onClose,
  onSave,
}: {
  initial: Combo | null;
  modelPairs: ModelRouterPair[];
  onClose: () => void;
  onSave: (c: Omit<Combo, "id"> & { id?: string }) => Promise<void>;
}) {
  const [c, setC] = useState<Omit<Combo, "id"> & { id?: string }>(
    initial ? { ...initial } : { ...DEFAULT_COMBO },
  );
  const set = <K extends keyof typeof c>(k: K, v: (typeof c)[K]) =>
    setC((prev) => ({ ...prev, [k]: v }));

  const imagePair = findPair(modelPairs, c.image_model_pair, c.image_model, DEFAULT_IMAGE_PAIR);
  const promptPair = findPair(modelPairs, c.prompt_model_pair, c.prompt_model, DEFAULT_PROMPT_PAIR);
  const imageThinking = validThinking(c.image_thinking, imagePair);
  const promptThinking = validThinking(c.prompt_thinking, promptPair);
  const canSave = c.name.trim().length > 0;

  const savePayload = {
    ...c,
    image_model_pair: imagePair,
    image_model: imagePair.model_id,
    image_thinking: imageThinking,
    prompt_model_pair: promptPair,
    prompt_model: promptPair.model_id,
    prompt_thinking: promptThinking,
  };

  return (
    <Modal
      title={initial ? "Edit Combo" : "New Combo"}
      onClose={onClose}
      footer={
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button onClick={onClose}>Cancel</button>
          <button className="primary" disabled={!canSave} onClick={() => onSave(savePayload)}>
            Save
          </button>
        </div>
      }
    >
      <div className="field">
        <label>Name</label>
        <input type="text" value={c.name} onChange={(e) => set("name", e.target.value)} placeholder="e.g. pro-max agent / 1:1" />
      </div>

      <div className="field">
        <label>Flow</label>
        <select value={c.flow} onChange={(e) => set("flow", e.target.value as FlowType)}>
          {FLOWS.map((f) => (
            <option key={f.id} value={f.id}>{f.label}</option>
          ))}
        </select>
      </div>

      <div className="grid cols-2">
        <div className="field">
          <label>Image model/router pair</label>
          <PairSelect
            value={imagePair.id}
            options={modelPairs.filter((p) => p.roles.includes("image"))}
            fallback={DEFAULT_IMAGE_PAIR}
            onChange={(pair) =>
              setC((prev) => ({
                ...prev,
                image_model_pair: pair,
                image_model: pair.model_id,
                image_thinking: pair.thinking_modes[0],
              }))
            }
          />
        </div>
        <div className="field">
          <label>Image thinking</label>
          <select value={imageThinking} onChange={(e) => set("image_thinking", e.target.value as ThinkingMode)}>
            {imagePair.thinking_modes.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Prompt/designer model/router pair</label>
          <PairSelect
            value={promptPair.id}
            options={modelPairs.filter((p) => p.roles.includes("prompt"))}
            fallback={DEFAULT_PROMPT_PAIR}
            onChange={(pair) =>
              setC((prev) => ({
                ...prev,
                prompt_model_pair: pair,
                prompt_model: pair.model_id,
                prompt_thinking: pair.thinking_modes[0],
              }))
            }
          />
        </div>
        <div className="field">
          <label>Prompt thinking</label>
          <select value={promptThinking} onChange={(e) => set("prompt_thinking", e.target.value as ThinkingMode)}>
            {promptPair.thinking_modes.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="field">
          <label>Max eval rounds</label>
          <input type="number" min={0} value={c.max_eval_rounds} onChange={(e) => set("max_eval_rounds", Number(e.target.value))} />
        </div>
        <div className="field">
          <label>Max prompt verify rounds</label>
          <input type="number" min={0} value={c.max_prompt_verify_rounds} onChange={(e) => set("max_prompt_verify_rounds", Number(e.target.value))} />
        </div>
      </div>

      <div className="field">
        <label>Aspect ratio (optional)</label>
        <input type="text" value={c.aspect_ratio ?? ""} placeholder="1:1, 16:9, auto" onChange={(e) => set("aspect_ratio", e.target.value || undefined)} />
      </div>

      {c.flow === "direct" && (
        <div className="field">
          <label>System prompt override (used by `direct` flow only)</label>
          <textarea value={c.system_prompt_override ?? ""} placeholder="Fixed system prompt for direct generation..." onChange={(e) => set("system_prompt_override", e.target.value || undefined)} style={{ minHeight: 120 }} />
        </div>
      )}

      {c.flow === "designer" && (
        <>
          <div className="field">
            <label>System prompt override (designer dev override text)</label>
            <textarea
              value={c.system_prompt_override ?? ""}
              placeholder="Optional dev override appended/replaced per mode below..."
              onChange={(e) => set("system_prompt_override", e.target.value || undefined)}
              style={{ minHeight: 100 }}
            />
          </div>
          <div className="field">
            <label>Designer system prompt mode</label>
            <select
              value={c.designer_system_prompt_mode ?? "append"}
              onChange={(e) =>
                set(
                  "designer_system_prompt_mode",
                  e.target.value as Combo["designer_system_prompt_mode"],
                )
              }
            >
              <option value="append">append</option>
              <option value="compose">compose</option>
              <option value="replace">replace</option>
            </select>
          </div>
          <div className="field">
            <label>Scoped filesystem root (optional)</label>
            <input
              type="text"
              value={c.designer_filesystem_root ?? ""}
              onChange={(e) => set("designer_filesystem_root", e.target.value || undefined)}
              placeholder="C:\path\to\scratch"
            />
          </div>
          <div className="field">
            <label>Max generations (budget)</label>
            <input
              type="number"
              min={1}
              value={c.designer_max_generation_rounds ?? 5}
              onChange={(e) =>
                set("designer_max_generation_rounds", e.target.value ? Number(e.target.value) : undefined)
              }
            />
          </div>
          <div className="field">
            <label>Max agent turns (0 = unlimited)</label>
            <input
              type="number"
              min={0}
              value={c.designer_max_turns ?? 24}
              onChange={(e) =>
                set(
                  "designer_max_turns",
                  e.target.value === "" ? undefined : Number(e.target.value),
                )
              }
            />
          </div>
          <div className="field">
            <label>Wall clock seconds (0 = unlimited, default 900)</label>
            <input
              type="number"
              min={0}
              value={c.designer_wall_clock_budget_s ?? 900}
              onChange={(e) =>
                set(
                  "designer_wall_clock_budget_s",
                  e.target.value === "" ? undefined : Number(e.target.value),
                )
              }
            />
          </div>
          <div className="field">
            <label>Designer tool access</label>
            <div className="row wrap">
              {[
                "ai_search",
                "keyword_search",
                "get_image",
                "generate_multi_image",
                "generate_multi_image_pipeline",
                "filesystem",
              ].map((tool) => (
                <label key={tool} className="check-label">
                  <input
                    type="checkbox"
                    checked={(c.designer_tool_access ?? []).includes(tool)}
                    onChange={() => {
                      const cur = new Set(c.designer_tool_access ?? []);
                      if (cur.has(tool)) cur.delete(tool);
                      else cur.add(tool);
                      set("designer_tool_access", Array.from(cur));
                    }}
                  />
                  {tool}
                </label>
              ))}
            </div>
          </div>
        </>
      )}
    </Modal>
  );
}

function PairSelect({
  value,
  options,
  fallback,
  onChange,
}: {
  value: string;
  options: ModelRouterPair[];
  fallback: ModelRouterPair;
  onChange: (v: ModelRouterPair) => void;
}) {
  const all = options.length ? options : [fallback];
  return (
    <select
      value={all.some((o) => o.id === value) ? value : all[0].id}
      onChange={(e) => onChange(all.find((o) => o.id === e.target.value) ?? all[0])}
    >
      {all.map((o) => (
        <option key={o.id} value={o.id}>{o.model_id} · {o.router}</option>
      ))}
    </select>
  );
}

function AddModelPair({
  routers,
  onSave,
}: {
  routers: RouterId[];
  onSave: (pair: { router: RouterId; model_id: string; label?: string }) => Promise<void>;
}) {
  const [router, setRouter] = useState<RouterId>("openrouter");
  const [modelId, setModelId] = useState("");
  const canSave = modelId.trim().length > 0;
  return (
    <div className="panel" style={{ marginBottom: 12 }}>
      <div className="spread">
        <div>
          <strong>Add model/router pair</strong>
          <div className="small muted">Use official IDs, e.g. <span className="mono">anthropic/claude-sonnet-4.5</span> for OpenRouter.</div>
        </div>
        <div className="row" style={{ minWidth: 520 }}>
          <select value={router} onChange={(e) => setRouter(e.target.value as RouterId)}>
            {routers.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
          <input type="text" value={modelId} placeholder="official model id" onChange={(e) => setModelId(e.target.value)} />
          <button
            disabled={!canSave}
            onClick={async () => {
              await onSave({ router, model_id: modelId.trim() });
              setModelId("");
            }}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

function findPair(
  pairs: ModelRouterPair[],
  saved: ModelRouterPair | undefined,
  legacyModel: string,
  fallback: ModelRouterPair,
): ModelRouterPair {
  if (saved) {
    return pairs.find((p) => p.id === saved.id) ?? saved;
  }
  return pairs.find((p) => p.model_id === legacyModel) ?? fallback;
}

function validThinking(value: ThinkingMode, pair: ModelRouterPair): ThinkingMode {
  return pair.thinking_modes.includes(value) ? value : pair.thinking_modes[0];
}
