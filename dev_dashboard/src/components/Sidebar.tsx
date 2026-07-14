import { useMemo, useState } from "react";
import { useStore, type Tab } from "../store";
import { catalogImageKey } from "../catalogUtils";

const TABS: { id: Tab; label: string }[] = [
  { id: "test-cases", label: "Test Cases" },
  { id: "combos", label: "Combinations" },
  { id: "designer-agent", label: "Designer Agent" },
  { id: "run-queue", label: "Run Queue" },
  { id: "live", label: "Live" },
  { id: "results", label: "Results / Compare" },
  { id: "history", label: "History" },
];

export default function Sidebar() {
  const activeTab = useStore((s) => s.activeTab);
  const setTab = useStore((s) => s.setTab);
  const backendUp = useStore((s) => s.backendUp);
  const backendError = useStore((s) => s.backendError);
  const catalogImages = useStore((s) => s.catalogImages);
  const [q, setQ] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");

  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    return catalogImages.filter((img) => {
      if (typeFilter !== "all" && img.type !== typeFilter) return false;
      if (!ql) return true;
      return (
        img.name?.toLowerCase().includes(ql) ||
        img.area?.toLowerCase().includes(ql) ||
        img.products?.some((p) => p.toLowerCase().includes(ql))
      );
    });
  }, [catalogImages, q, typeFilter]);

  const types = useMemo(() => {
    const s = new Set<string>();
    catalogImages.forEach((i) => s.add(i.type));
    return Array.from(s);
  }, [catalogImages]);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>Ducon Dev Dashboard</h1>
        <div className="sub">INTERNAL — NOT FOR PRODUCTION</div>
        <div className="small" style={{ marginTop: 6 }}>
          {backendUp ? (
            <span className="badge completed">● backend connected</span>
          ) : (
            <span className="badge failed" title={backendError ?? undefined}>
              ● backend offline
            </span>
          )}
        </div>
      </div>

      <nav className="sidebar-nav">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={activeTab === t.id ? "active" : ""}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-catalog">
        <h2>Image Catalog</h2>
        <div className="catalog-search">
          <input
            type="search"
            placeholder="search name / area / product…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <div className="row" style={{ marginTop: 6 }}>
            <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
              <option value="all">all types</option>
              {types.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <span className="small muted">{filtered.length}</span>
          </div>
        </div>
        <div className="catalog-list">
          {filtered.length === 0 && (
            <div className="muted small" style={{ gridColumn: "1 / -1", padding: 12 }}>
              {catalogImages.length === 0
                ? "No catalog loaded (backend offline?)."
                : "No matches."}
            </div>
          )}
          {filtered.map((img, i) => (
            <div
              key={catalogImageKey(img, i)}
              className="catalog-item"
              title={`${img.name} (${img.type})`}
              onClick={() => setTab("test-cases")}
            >
              <img
                src={img.thumb_url || img.full_url}
                alt={img.name}
                className={img.thumb_url ? "" : "missing"}
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).classList.add("missing");
                }}
              />
              <div className="name">{img.name}</div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
