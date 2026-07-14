import { useEffect, useMemo } from "react";
import { useStore } from "../store";
import { Empty, fmtCost, fmtMs, statusBadge } from "../components/ui";

export default function Results() {
  const runGroup = useStore((s) => s.runGroup);
  const processes = useStore((s) => s.processes);
  const testCases = useStore((s) => s.testCases);
  const combos = useStore((s) => s.combos);
  const selectProcess = useStore((s) => s.selectProcess);
  const setTab = useStore((s) => s.setTab);
  const refreshRunSnapshot = useStore((s) => s.refreshRunSnapshot);

  const tcById = useMemo(() => new Map(testCases.map((t) => [t.id, t])), [testCases]);
  const coById = useMemo(() => new Map(combos.map((c) => [c.id, c])), [combos]);

  // Keep snapshot fresh so the benchmark view reflects final results.
  useEffect(() => {
    if (!runGroup) return;
    refreshRunSnapshot(runGroup.id);
    const id = setInterval(() => refreshRunSnapshot(runGroup.id), 8000);
    return () => clearInterval(id);
  }, [runGroup, refreshRunSnapshot]);

  const rows = useMemo(
    () => Object.values(processes),
    [processes],
  );

  if (!runGroup) {
    return (
      <Empty>
        No run loaded. Start one from{" "}
        <button className="ghost" onClick={() => setTab("run-queue")}>Run Queue</button>{" "}
        or open one from{" "}
        <button className="ghost" onClick={() => setTab("history")}>History</button>.
      </Empty>
    );
  }

  return (
    <div>
      <div className="spread" style={{ marginBottom: 12 }}>
        <div className="row">
          <h2 style={{ margin: 0 }}>Results / Compare</h2>
          <span className="muted small">run group {runGroup.id}</span>
        </div>
        <div className="row">
          <button onClick={() => refreshRunSnapshot(runGroup.id)}>Refresh</button>
          <button onClick={() => setTab("live")}>← Live</button>
        </div>
      </div>

      {rows.length === 0 ? (
        <Empty>No processes.</Empty>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Combo</th>
              <th>Test case</th>
              <th>Status</th>
              <th>Total time</th>
              <th>Retries</th>
              <th>Total cost</th>
              <th>Output</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => {
              const out = p.result?.output_images?.[0]?.url;
              const dur =
                p.result?.total_duration_ms ??
                (p.started_at
                  ? (p.ended_at ? new Date(p.ended_at).getTime() : Date.now()) -
                    new Date(p.started_at).getTime()
                  : undefined);
              return (
                <tr
                  key={p.id}
                  className="clickable"
                  onClick={() => selectProcess(p.id)}
                >
                  <td>{coById.get(p.combo_id)?.name ?? p.combo_id}</td>
                  <td>{tcById.get(p.test_case_id)?.name ?? p.test_case_id}</td>
                  <td>{statusBadge(p.status)}</td>
                  <td className="mono small">{fmtMs(dur)}</td>
                  <td className="mono small">{p.result?.retries ?? 0}</td>
                  <td className="mono small">{fmtCost(p.result?.total_cost_usd)}</td>
                  <td>
                    {out ? (
                      <img className="thumb" src={out} alt="output" />
                    ) : (
                      <span className="muted small">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
