import { useEffect, useMemo } from "react";
import { useStore } from "../store";
import { Empty, statusBadge } from "../components/ui";

export default function History() {
  const history = useStore((s) => s.history);
  const designerHistory = useStore((s) => s.designerHistory);
  const refreshHistory = useStore((s) => s.refreshHistory);
  const deleteRunGroup = useStore((s) => s.deleteRunGroup);
  const deleteDesignerJob = useStore((s) => s.deleteDesignerJob);
  const refreshRunSnapshot = useStore((s) => s.refreshRunSnapshot);
  const loadRun = useStore((s) => s.loadRun);
  const selectDesignerJob = useStore((s) => s.selectDesignerJob);
  const setTab = useStore((s) => s.setTab);
  const backendUp = useStore((s) => s.backendUp);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  const rows = useMemo(() => {
    const benchmark = history.map((h) => ({
      id: h.id,
      kind: "benchmark" as const,
      created_at: h.created_at,
      label: h.id,
      detail: `${h.test_case_count} test cases · ${h.combo_count} combos`,
      status: h.status,
    }));
    const designer = designerHistory.map((j) => ({
      id: j.id,
      kind: "designer" as const,
      created_at: j.created_at,
      label: j.id,
      detail: (j.prompt || j.upload_name || "Designer job").slice(0, 80),
      status: j.status,
    }));
    return [...benchmark, ...designer].sort((a, b) => {
      const ta = new Date(a.created_at).getTime();
      const tb = new Date(b.created_at).getTime();
      return tb - ta;
    });
  }, [history, designerHistory]);

  return (
    <div>
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>History</h2>
        <button onClick={() => refreshHistory()}>Refresh</button>
      </div>

      {!backendUp && (
        <div className="banner err">
          Backend not connected — history is only available when the backend is up.
        </div>
      )}

      {rows.length === 0 ? (
        <Empty>No past runs or designer jobs on the backend.</Empty>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Kind</th>
              <th>ID</th>
              <th>Created</th>
              <th>Summary</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((h) => (
              <tr key={`${h.kind}-${h.id}`}>
                <td>
                  <span className="tag-pill">{h.kind}</span>
                </td>
                <td className="mono small">{h.id}</td>
                <td className="small">{new Date(h.created_at).toLocaleString()}</td>
                <td className="small">{h.detail}</td>
                <td>{statusBadge(h.status as import("../types").ProcessStatus)}</td>
                <td>
                  <div className="row" style={{ justifyContent: "flex-end" }}>
                    <button
                      onClick={async () => {
                        if (h.kind === "benchmark") {
                          await refreshRunSnapshot(h.id);
                          await loadRun(h.id);
                          setTab("results");
                        } else {
                          selectDesignerJob(h.id);
                        }
                      }}
                    >
                      Open →
                    </button>
                    <button
                      className="ghost"
                      onClick={async () => {
                        if (!window.confirm(`Delete ${h.kind} ${h.id}?`)) return;
                        if (h.kind === "benchmark") await deleteRunGroup(h.id);
                        else await deleteDesignerJob(h.id);
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
    </div>
  );
}
