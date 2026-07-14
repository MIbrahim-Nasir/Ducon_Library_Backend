import { useEffect, useState } from "react";
import { create } from "zustand";
import { api, ApiError } from "../api";

interface LightboxState {
  url: string | null;
  open: (url: string) => void;
  close: () => void;
}

// Small standalone store so image click handlers anywhere can open the lightbox
// without subscribing to (and re-rendering on) the main app store.
export const useLightbox = create<LightboxState>((set) => ({
  url: null,
  open: (url) => set({ url }),
  close: () => set({ url: null }),
}));

/** Open the lightbox from anywhere without subscribing to the store. */
export function openLightbox(url: string) {
  useLightbox.getState().open(url);
}

export default function Lightbox() {
  const url = useLightbox((s) => s.url);
  const close = useLightbox((s) => s.close);
  const [status, setStatus] = useState<{ kind: "idle" | "loading" | "ok" | "err"; msg?: string }>({
    kind: "idle",
  });

  useEffect(() => {
    if (!url) return;
    setStatus({ kind: "idle" });
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [url, close]);

  if (!url) return null;

  const reveal = async () => {
    setStatus({ kind: "loading" });
    try {
      const r = await api.reveal(url);
      if (r.ok) {
        setStatus({ kind: "ok", msg: r.path ? `Revealed: ${r.path}` : "Revealed in folder." });
      } else {
        setStatus({ kind: "err", msg: r.error || "Failed to reveal file." });
      }
    } catch (e) {
      setStatus({ kind: "err", msg: e instanceof ApiError ? e.message : String(e) });
    }
  };

  return (
    <div className="lightbox-backdrop" onClick={close}>
      <div className="lightbox" onClick={(e) => e.stopPropagation()}>
        <button className="lightbox-close" onClick={close} aria-label="Close">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <path
              d="M2 2 L12 12 M12 2 L2 12"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
        </button>
        <img className="lightbox-img" src={url} alt="Enlarged image" />
        <div className="lightbox-actions">
          <button onClick={reveal} disabled={status.kind === "loading"}>
            {status.kind === "loading" ? "Opening…" : "Open in folder"}
          </button>
          {status.kind === "ok" && (
            <span className="lightbox-status ok">{status.msg}</span>
          )}
          {status.kind === "err" && (
            <span className="lightbox-status err">{status.msg}</span>
          )}
        </div>
      </div>
    </div>
  );
}
