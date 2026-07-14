import type { ReactNode } from "react";

export function Modal({
  title,
  onClose,
  children,
  footer,
}: {
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <strong>{title}</strong>
          <button className="ghost" onClick={onClose} aria-label="close">
            ✕
          </button>
        </header>
        <div className="body">{children}</div>
        {footer && <footer>{footer}</footer>}
      </div>
    </div>
  );
}

export function Banner({
  kind,
  children,
}: {
  kind: "ok" | "err";
  children: ReactNode;
}) {
  return <div className={`banner ${kind}`}>{children}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

// Formats ms as "1m 23s" / "12.3s" / "—".
export function fmtMs(ms?: number): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rs = Math.round(s % 60);
  return `${m}m ${rs}s`;
}

export function fmtCost(c?: number): string {
  if (c == null) return "—";
  if (c < 0.01) return `$${c.toFixed(5)}`;
  return `$${c.toFixed(4)}`;
}

export function statusBadge(status: string) {
  return <span className={`badge ${status}`}>{status}</span>;
}
