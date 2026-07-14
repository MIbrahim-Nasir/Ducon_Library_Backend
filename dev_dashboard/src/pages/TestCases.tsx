import { useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import { api } from "../api";
import { catalogImageKey } from "../catalogUtils";
import { Banner, Empty, Modal, statusBadge } from "../components/ui";
import type { TestCase, TestCaseImage } from "../types";

export default function TestCases() {
  const testCases = useStore((s) => s.testCases);
  const backendUp = useStore((s) => s.backendUp);
  const saveTestCase = useStore((s) => s.saveTestCase);
  const deleteTestCase = useStore((s) => s.deleteTestCase);
  const refreshTestCases = useStore((s) => s.refreshTestCases);

  const [editing, setEditing] = useState<TestCase | null>(null);
  const [creating, setCreating] = useState(false);

  return (
    <div>
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Test Cases</h2>
        <div className="row">
          <button onClick={() => refreshTestCases()}>Refresh</button>
          <button
            className="primary"
            onClick={() => {
              setEditing(null);
              setCreating(true);
            }}
          >
            + New Test Case
          </button>
        </div>
      </div>

      {!backendUp && (
        <Banner kind="err">
          Backend not connected — test cases are read from local cache only.
          Creators still work and are persisted to localStorage; they will sync
          to the backend when it comes back online.
        </Banner>
      )}

      {testCases.length === 0 ? (
        <Empty>
          No test cases yet. Click <strong>+ New Test Case</strong> to create
          one. Each input image bundle you save is a reusable test case.
        </Empty>
      ) : (
        <div className="grid cols-3">
          {testCases.map((tc) => (
            <div key={tc.id} className="card">
              <div className="spread">
                <strong>{tc.name}</strong>
                {tc.use_ducon_data && <span className="tag-pill">ducon data</span>}
              </div>
              <div className="image-strip">
                {(tc.user_images ?? []).map((im, i) => (
                  <div className="image-tile" key={`u${i}`}>
                    <img src={im.url} alt="" />
                    <div className="tag">user</div>
                  </div>
                ))}
                {(tc.ducon_images ?? []).map((im, i) => (
                  <div className="image-tile" key={`d${i}`}>
                    <img src={im.url} alt="" />
                    <div className="tag">ducon</div>
                  </div>
                ))}
              </div>
              {tc.hint && (
                <div className="small muted" style={{ marginTop: 8 }}>
                  <em>hint:</em> {tc.hint}
                </div>
              )}
              <div className="row" style={{ marginTop: 10, justifyContent: "flex-end" }}>
                <button
                  onClick={() => {
                    setEditing(tc);
                    setCreating(true);
                  }}
                >
                  Edit
                </button>
                <button
                  className="danger"
                  onClick={() => {
                    if (confirm(`Delete test case "${tc.name}"?`)) deleteTestCase(tc.id);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {creating && (
        <TestCaseEditor
          initial={editing}
          onClose={() => setCreating(false)}
          onSave={async (tc) => {
            await saveTestCase(tc);
            setCreating(false);
          }}
        />
      )}
    </div>
  );
}

function TestCaseEditor({
  initial,
  onClose,
  onSave,
}: {
  initial: TestCase | null;
  onClose: () => void;
  onSave: (tc: Omit<TestCase, "id"> & { id?: string }) => Promise<void>;
}) {
  const catalogImages = useStore((s) => s.catalogImages);
  const catalogProducts = useStore((s) => s.catalogProducts);
  const catalogAreas = useStore((s) => s.catalogAreas);
  const backendUp = useStore((s) => s.backendUp);

  const [name, setName] = useState(initial?.name ?? "");
  const [userImages, setUserImages] = useState<TestCaseImage[]>(
    initial?.user_images ?? [],
  );
  const [duconImages, setDuconImages] = useState<TestCaseImage[]>(
    initial?.ducon_images ?? [],
  );
  const [useDucon, setUseDucon] = useState(initial?.use_ducon_data ?? true);
  const [hint, setHint] = useState(initial?.hint ?? "");
  const [uploading, setUploading] = useState(false);
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [catQ, setCatQ] = useState("");
  const [catType, setCatType] = useState<string>("all");
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const filteredCatalog = useMemo(() => {
    const ql = catQ.trim().toLowerCase();
    return catalogImages.filter((i) => {
      if (catType !== "all" && i.type !== catType) return false;
      if (!ql) return true;
      return (
        i.name?.toLowerCase().includes(ql) ||
        i.area?.toLowerCase().includes(ql) ||
        i.products?.some((p) => p.toLowerCase().includes(ql))
      );
    });
  }, [catalogImages, catQ, catType]);

  const catTypes = useMemo(() => {
    const s = new Set(catalogImages.map((i) => i.type));
    return Array.from(s);
  }, [catalogImages]);

  async function handleFiles(files: FileList | File[]) {
    setUploading(true);
    try {
      for (const f of Array.from(files)) {
        let url: string;
        if (backendUp) {
          try {
            const up = await api.upload(f);
            url = up.url;
            setUserImages((prev) => [
              ...prev,
              {
                url,
                role: "user",
                metadata: { filename: f.name, size: f.size, upload_id: up.id },
              },
            ]);
            continue;
          } catch {
            // Fallback to a local object URL so the test case is still usable.
            url = URL.createObjectURL(f);
          }
        } else {
          url = URL.createObjectURL(f);
        }
        setUserImages((prev) => [
          ...prev,
          { url, role: "user", metadata: { filename: f.name, size: f.size } },
        ]);
      }
    } finally {
      setUploading(false);
    }
  }

  function addDucon(img: (typeof catalogImages)[number]) {
    setDuconImages((prev) => [
      ...prev,
      {
        url: img.full_url || img.thumb_url,
        role: "ducon",
        metadata: {
          id: img.id,
          catalog_id: img.id,
          type: img.type,
          area: img.area,
          products: img.products,
          name: img.name,
          filename: img.filename,
          full_url: img.full_url || img.thumb_url,
          thumb_url: img.thumb_url,
        },
      },
    ]);
  }

  const canSave = name.trim().length > 0 && (userImages.length > 0 || duconImages.length > 0);

  return (
    <Modal
      title={initial ? "Edit Test Case" : "New Test Case"}
      onClose={onClose}
      footer={
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button onClick={onClose}>Cancel</button>
          <button
            className="primary"
            disabled={!canSave}
            onClick={() =>
              onSave({
                id: initial?.id,
                name: name.trim(),
                user_images: userImages,
                ducon_images: duconImages,
                use_ducon_data: useDucon,
                hint: hint.trim() || undefined,
              })
            }
          >
            Save
          </button>
        </div>
      }
    >
      <div className="field">
        <label>Name</label>
        <input
          type="text"
          value={name}
          placeholder="e.g. Living room with sectional sofa"
          onChange={(e) => setName(e.target.value)}
        />
      </div>

      <div className="field">
        <label>User images (upload)</label>
        <div
          className={`dropzone ${dragOver ? "drag" : ""}`}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
          }}
        >
          {uploading
            ? "Uploading…"
            : "Drag & drop images here, or click to pick files"}
        </div>
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          multiple
          style={{ display: "none" }}
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
        {userImages.length > 0 && (
          <div className="image-strip">
            {userImages.map((im, i) => (
              <div className="image-tile" key={`u${i}`}>
                <img src={im.url} alt="" />
                <button
                  className="rm"
                  onClick={() =>
                    setUserImages((p) => p.filter((_, idx) => idx !== i))
                  }
                >
                  ✕
                </button>
                <div className="tag">{(im.metadata?.filename as string) ?? "user"}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="field">
        <label>Ducon library images</label>
        <button onClick={() => setCatalogOpen((v) => !v)}>
          {catalogOpen ? "Hide catalog browser" : "Browse catalog"}
        </button>
        {catalogOpen && (
          <div style={{ marginTop: 8, border: "1px solid var(--border)", borderRadius: 6, padding: 8 }}>
            <div className="row" style={{ marginBottom: 8 }}>
              <input
                type="search"
                placeholder="search catalog…"
                value={catQ}
                onChange={(e) => setCatQ(e.target.value)}
              />
              <select value={catType} onChange={(e) => setCatType(e.target.value)}>
                <option value="all">all</option>
                {catTypes.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
            {catalogImages.length === 0 && (
              <div className="small muted">
                Catalog empty — backend offline or no images ingested.
                {catalogAreas.length > 0 || catalogProducts.length > 0
                  ? " (areas/products available though.)"
                  : ""}
              </div>
            )}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))",
                gap: 6,
                maxHeight: 260,
                overflowY: "auto",
              }}
            >
              {filteredCatalog.map((img, i) => (
                <div
                  key={catalogImageKey(img, i)}
                  className="catalog-item"
                  onClick={() => addDucon(img)}
                  title={`Add ${img.name}`}
                >
                  <img src={img.thumb_url || img.full_url} alt={img.name} />
                  <div className="name">+ {img.name}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {duconImages.length > 0 && (
          <div className="image-strip">
            {duconImages.map((im, i) => (
              <div className="image-tile" key={`d${i}`}>
                <img src={im.url} alt="" />
                <button
                  className="rm"
                  onClick={() =>
                    setDuconImages((p) => p.filter((_, idx) => idx !== i))
                  }
                >
                  ✕
                </button>
                <div className="tag">{(im.metadata?.type as string) ?? "ducon"}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="field">
        <label>
          <input
            type="checkbox"
            className="checkbox"
            checked={useDucon}
            onChange={(e) => setUseDucon(e.target.checked)}
            style={{ width: "auto", marginRight: 6 }}
          />
          use_ducon_data — feed selected ducon catalog data into generation
        </label>
      </div>

      <div className="field">
        <label>Optional hint / prompt</label>
        <textarea
          value={hint}
          placeholder="e.g. warm minimalist, keep the existing wall color"
          onChange={(e) => setHint(e.target.value)}
        />
      </div>
    </Modal>
  );
}
