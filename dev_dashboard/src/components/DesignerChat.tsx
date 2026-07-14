import { useState, type MouseEvent } from "react";
import type { DesignerJobEvent, DesignerJobFinal } from "../types";
import { api, ApiError } from "../api";
import { openLightbox } from "./Lightbox";

interface DesignerChatProps {
  uploadUrl?: string;
  uploadName?: string;
  prompt?: string;
  events: DesignerJobEvent[];
  final?: DesignerJobFinal | null;
  running?: boolean;
}

export default function DesignerChat({
  uploadUrl,
  uploadName,
  prompt,
  events,
  final,
  running,
}: DesignerChatProps) {
  const messages = buildMessages({ uploadUrl, uploadName, prompt, events, final });

  return (
    <div className="designer-chat">
      {messages.length === 0 && !running ? (
        <p className="muted small">Run the designer agent to see the conversation here.</p>
      ) : (
        messages.map((msg, idx) => (
          <div key={idx} className={`chat-msg chat-msg-${msg.role}`}>
            <div className="chat-msg-head">
              <strong>{msg.label}</strong>
              {msg.sub && <span className="muted small">{msg.sub}</span>}
            </div>
            {msg.imageUrl && (
              <div className="chat-upload">
                <ChatImage src={msg.imageUrl} alt={msg.imageAlt || "upload"} />
                {msg.imageAlt && <span className="small muted">{msg.imageAlt}</span>}
              </div>
            )}
            {msg.text && <div className="chat-text">{msg.text}</div>}
            {msg.toolName && (
              <div className="chat-tool">
                <span className="tag-pill">{msg.toolName}</span>
                {msg.toolDetail && <pre className="chat-tool-detail">{msg.toolDetail}</pre>}
              </div>
            )}
            {msg.images && msg.images.length > 0 && (
              <div className="result-images">
                {msg.images.map((img) => (
                  <ChatImage key={img.url} src={img.url} alt="generated output" />
                ))}
              </div>
            )}
          </div>
        ))
      )}
      {running && (
        <div className="chat-msg chat-msg-assistant">
          <div className="chat-msg-head">
            <strong>Designer</strong>
            <span className="muted small">running…</span>
          </div>
        </div>
      )}
    </div>
  );
}

interface ChatMessage {
  role: "user" | "assistant" | "tool";
  label: string;
  sub?: string;
  text?: string;
  imageUrl?: string;
  imageAlt?: string;
  toolName?: string;
  toolDetail?: string;
  images?: { url: string }[];
}

/** A chat image: click to open the lightbox, hover for an "Open in folder" button. */
function ChatImage({ src, alt }: { src: string; alt: string }) {
  const [state, setState] = useState<"idle" | "loading" | "ok" | "err">("idle");
  const [err, setErr] = useState<string | null>(null);

  const reveal = async (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (state === "loading") return;
    setState("loading");
    try {
      const r = await api.reveal(src);
      if (r.ok) {
        setState("ok");
        setErr(null);
      } else {
        setState("err");
        setErr(r.error || "Failed to reveal file.");
      }
    } catch (e) {
      setState("err");
      setErr(e instanceof ApiError ? e.message : String(e));
    }
  };

  return (
    <span className="img-wrap">
      <img src={src} alt={alt} onClick={() => openLightbox(src)} />
      <button
        type="button"
        className="img-actions"
        title="Open in folder"
        onClick={reveal}
        disabled={state === "loading"}
      >
        {state === "loading" ? "Opening…" : state === "ok" ? "Opened" : "Open in folder"}
      </button>
      {state === "err" && err && <span className="img-actions-err">{err}</span>}
    </span>
  );
}

/** Render a round index verbatim (backend convention-agnostic). */
function roundLabel(round: unknown): string {
  if (typeof round === "number" && Number.isFinite(round)) return String(round);
  if (round != null) return String(round);
  return "?";
}

function buildMessages({
  uploadUrl,
  uploadName,
  prompt,
  events,
  final,
}: {
  uploadUrl?: string;
  uploadName?: string;
  prompt?: string;
  events: DesignerJobEvent[];
  final?: DesignerJobFinal | null;
}): ChatMessage[] {
  const out: ChatMessage[] = [];

  // Emit the user's input message EXACTLY ONCE, up front, from whatever the
  // session knows (uploadUrl/uploadName/prompt). The loop below never pushes a
  // second user message — an `input_image` event only refines these values in
  // place. This kills the double user-message bug where a non-input_image event
  // arriving before `input_image` triggered the in-loop fallback push AND then
  // `input_image` pushed again.
  if (uploadUrl || (prompt && prompt.trim())) {
    out.push({
      role: "user",
      label: "You",
      text: prompt?.trim() || undefined,
      imageUrl: uploadUrl,
      imageAlt: uploadName || "Uploaded space photo",
    });
  }

  for (const ev of events) {
    if (ev.type === "input_image") {
      // Refine the leading user message with richer info from the event, or
      // insert one if the session had no upload/prompt to seed it. Never append.
      const refined: ChatMessage = {
        role: "user",
        label: "You",
        text: (String(ev.prompt || "") || prompt || "").trim() || undefined,
        imageUrl: String(ev.url || "") || uploadUrl,
        imageAlt: String(ev.name || "") || uploadName || "Uploaded space photo",
      };
      if (out[0]?.role === "user") {
        out[0] = refined;
      } else {
        out.unshift(refined);
      }
      continue;
    }

    switch (ev.type) {
      case "assistant_message":
        out.push({
          role: "assistant",
          label: "Designer",
          sub: ev.turn != null ? `turn ${String(ev.turn)}` : undefined,
          text: String(ev.text || ""),
        });
        break;
      case "tool_call": {
        const name = String(ev.name || "tool");
        const args = ev.args && typeof ev.args === "object" ? ev.args : {};
        out.push({
          role: "tool",
          label: "Tool call",
          sub: ev.turn != null ? `turn ${String(ev.turn)}` : undefined,
          toolName: name,
          toolDetail: JSON.stringify(args, null, 2).slice(0, 4000),
        });
        break;
      }
      case "tool_result": {
        const name = String(ev.name || "tool");
        const result = ev.result && typeof ev.result === "object" ? (ev.result as Record<string, unknown>) : {};
        const failed = Boolean(result.error);
        const images = Array.isArray(ev.output_images)
          ? (ev.output_images as { url: string }[])
          : [];
        out.push({
          role: "tool",
          label: failed ? "Tool error" : "Tool result",
          sub: ev.turn != null ? `turn ${String(ev.turn)}` : undefined,
          toolName: `${name} — ${failed ? "error" : "done"}`,
          toolDetail: JSON.stringify(result, null, 2).slice(0, 4000),
          images,
        });
        break;
      }
      case "status":
        out.push({
          role: "assistant",
          label: "Designer",
          text: String(ev.message || "Working…"),
        });
        break;
      case "plan":
        out.push({
          role: "assistant",
          label: "Designer",
          sub: "Design plan",
          toolName: "plan",
          toolDetail: JSON.stringify(ev.plan, null, 2),
        });
        break;
      case "system_prompt_used":
        out.push({
          role: "tool",
          label: "System",
          toolName: `system_prompt (${String(ev.mode || "append")})`,
          toolDetail: String(ev.prompt || "").slice(0, 6000),
        });
        break;
      case "search_started":
        out.push({
          role: "tool",
          label: "Tool",
          toolName: `${String(ev.search_type || "search")} — started`,
          toolDetail: String(ev.query || ""),
        });
        break;
      case "search_done":
        out.push({
          role: "tool",
          label: "Tool",
          toolName: `${String(ev.search_type || "search")} — done`,
          toolDetail: `${String(ev.query || "")}\nHits: ${String(ev.hit_count ?? (Array.isArray(ev.raw_ids) ? ev.raw_ids.length : "?"))}`,
        });
        break;
      case "sources":
        out.push({
          role: "tool",
          label: "Tool",
          toolName: "sources",
          toolDetail: JSON.stringify(
            { sources: ev.sources, removed: ev.removed },
            null,
            2,
          ),
        });
        break;
      case "get_image":
        out.push({
          role: "tool",
          label: "Tool",
          toolName: "get_image",
          toolDetail: `Loaded ${String(ev.loaded_count ?? 0)} catalog image(s)\nIDs: ${JSON.stringify(ev.source_ids ?? [])}`,
        });
        break;
      case "tool_error":
        out.push({
          role: "tool",
          label: "Tool error",
          toolName: String(ev.tool || "tool"),
          toolDetail: String(ev.error || "Unknown error"),
        });
        break;
      case "generation_round_started":
        out.push({
          role: "assistant",
          label: "Designer",
          text: `Starting generation round ${roundLabel(ev.round)}`,
        });
        break;
      case "generation_started":
        out.push({
          role: "assistant",
          label: "Designer",
          text: String(ev.message || "Generating design candidate…"),
        });
        break;
      case "generation_done":
        // Each generation_done renders as its own assistant message with its
        // images, so multi-round refinement shows as a sequence of results.
        out.push({
          role: "assistant",
          label: "Designer",
          sub: ev.round != null ? `Generation complete (round ${roundLabel(ev.round)})` : "Generation complete",
          images: (ev.output_images as { url: string }[] | undefined) ?? [],
        });
        break;
      case "pipeline_step": {
        const kind = String(ev.kind || "step");
        const status = String(ev.status || "");
        const promptUsed = ev.prompt_used ? String(ev.prompt_used).slice(0, 2000) : "";
        out.push({
          role: "tool",
          label: "Pipeline",
          sub: ev.round != null ? `round ${roundLabel(ev.round)}` : undefined,
          toolName: `${kind} — ${status}`,
          toolDetail: promptUsed || undefined,
        });
        break;
      }
      case "session_compacted": {
        const actions = Array.isArray(ev.actions) ? ev.actions.join(", ") : "";
        out.push({
          role: "tool",
          label: "Session",
          sub: ev.turn != null ? `turn ${String(ev.turn)}` : undefined,
          toolName: "context compacted",
          toolDetail: [
            actions ? `Actions: ${actions}` : null,
            ev.estimated_tokens != null ? `Est. tokens: ${String(ev.estimated_tokens)}` : null,
            ev.messages_dropped != null ? `Messages dropped: ${String(ev.messages_dropped)}` : null,
            ev.summarize_error ? `Summarize error: ${String(ev.summarize_error)}` : null,
          ]
            .filter(Boolean)
            .join("\n"),
        });
        break;
      }
      case "eval": {
        const approved = ev.approved === true;
        const reasons = Array.isArray(ev.reasons) ? ev.reasons.filter(Boolean) : [];
        const defects = Array.isArray(ev.defects) ? ev.defects.filter(Boolean) : [];
        const rl = roundLabel(ev.round);
        const detail = [
          `Round: ${rl}`,
          `Approved: ${approved ? "yes" : "no"}`,
          reasons.length ? `Reasons:\n- ${reasons.join("\n- ")}` : null,
          defects.length ? `Defects:\n- ${defects.join("\n- ")}` : null,
        ]
          .filter(Boolean)
          .join("\n");
        out.push({
          role: "tool",
          label: "Evaluator",
          sub: `Round ${rl} — ${approved ? "approved" : "rejected"}`,
          toolName: approved ? "eval — approved" : "eval — rejected",
          toolDetail: detail,
        });
        break;
      }
      case "final":
        out.push({
          role: "assistant",
          label: "Designer",
          sub: "Final result",
          images:
            (ev.design_generation as { output_images?: { url: string }[] } | undefined)
              ?.output_images ?? final?.design_generation?.output_images?.map((i) => ({ url: i.url })),
        });
        break;
      case "error":
        out.push({
          role: "assistant",
          label: "Designer",
          text: String(ev.message || "Job failed."),
        });
        break;
      case "cancelled":
        out.push({
          role: "assistant",
          label: "Designer",
          text: String(ev.message || "Job cancelled."),
        });
        break;
      default:
        break;
    }
  }

  return out;
}
