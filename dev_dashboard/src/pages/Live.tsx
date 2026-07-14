import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store";
import { Banner, Empty, fmtMs, statusBadge } from "../components/ui";
import { openLightbox } from "../components/Lightbox";
import type { DesignerHistoryEntry, Process } from "../types";

export default function Live() {
  const runGroup = useStore((s) => s.runGroup);
  const processes = useStore((s) => s.processes);
  const testCases = useStore((s) => s.testCases);
  const combos = useStore((s) => s.combos);
  const designerSession = useStore((s) => s.designerSession);
  const designerHistory = useStore((s) => s.designerHistory);
  const selectProcess = useStore((s) => s.selectProcess);
  const selectDesignerJob = useStore((s) => s.selectDesignerJob);
  const setTab = useStore((s) => s.setTab);
  const refreshRunSnapshot = useStore((s) => s.refreshRunSnapshot);
  const refreshHistory = useStore((s) => s.refreshHistory);
  const streamConnected = useStore((s) => s.streamConnected);
  const loadRun = useStore((s) => s.loadRun);
  const closeStream = useStore((s) => s.closeStream);
  const cancelDesignerJob = useStore((s) => s.cancelDesignerJob);

  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!runGroup) return;
    const id = setInterval(() => refreshRunSnapshot(runGroup.id), 5000);
    return () => clearInterval(id);
  }, [runGroup, refreshRunSnapshot]);

  useEffect(() => {
    const id = setInterval(() => refreshHistory(), 5000);
    return () => clearInterval(id);
  }, [refreshHistory]);

  const procList = useMemo(() => Object.values(processes), [processes]);
  const tcById = useMemo(() => new Map(testCases.map((t) => [t.id, t])), [testCases]);
  const coById = useMemo(() => new Map(combos.map((c) => [c.id, c])), [combos]);

  const activeDesignerJobs = useMemo(() => {
    const rows: DesignerHistoryEntry[] = [];
    if (designerSession.jobId && (designerSession.running || designerSession.events.length)) {
      rows.push({
        id: designerSession.jobId,
        kind: "designer",
        created_at: Date.now() / 1000,
        status: designerSession.running ? "running" : designerSession.error ? "failed" : "completed",
        prompt: designerSession.prompt,
        upload_url: designerSession.uploadUrl,
        upload_name: designerSession.uploadName,
        error: designerSession.error ?? undefined,
      });
    }
    for (const j of designerHistory) {
      if (j.status === "running" || j.status === "queued") {
        if (!rows.some((r) => r.id === j.id)) rows.push(j);
      }
    }
    return rows;
  }, [designerSession, designerHistory]);

  const allProcDone =
    procList.length > 0 &&
    procList.every((p) => p.status === "completed" || p.status === "failed");

  if (!runGroup && activeDesignerJobs.length === 0) {
    return (
      <Empty>
        No active runs. Start a{" "}
        <button className="ghost" onClick={() => setTab("run-queue")}>
          benchmark run
        </button>{" "}
        or a{" "}
        <button className="ghost" onClick={() => setTab("designer-agent")}>
          designer agent
        </button>{" "}
        job.
      </Empty>
    );
  }

  return (
    <div>
      {runGroup && (
        <>
          <div className="spread" style={{ marginBottom: 12 }}>
            <div className="row">
              <h2 style={{ margin: 0 }}>Live — Benchmark</h2>
              <span className="muted small">run group {runGroup.id}</span>
              {streamConnected ? (
                <span className="badge completed">● streaming</span>
              ) : (
                <span className="badge pending">○ stream idle</span>
              )}
            </div>
            <div className="row">
              <button onClick={() => loadRun(runGroup.id)}>Reconnect SSE</button>
              <button onClick={() => refreshRunSnapshot(runGroup.id)}>Refresh</button>
              <button
                onClick={() => {
                  closeStream();
                  setTab("results");
                }}
              >
                View Results →
              </button>
            </div>
          </div>

          {!streamConnected && (
            <Banner kind="err">
              SSE stream not connected. Polling run snapshot every 5s as fallback.
            </Banner>
          )}

          {procList.length === 0 ? (
            <Empty>No processes in this run group.</Empty>
          ) : (
            <div className="process-grid">
              {procList.map((p) => (
                <ProcessCard
                  key={p.id}
                  p={p}
                  tcName={tcById.get(p.test_case_id)?.name ?? p.test_case_id}
                  coName={coById.get(p.combo_id)?.name ?? p.combo_id}
                  onOpen={() => selectProcess(p.id)}
                />
              ))}
            </div>
          )}

          {allProcDone && (
            <div style={{ marginTop: 16 }}>
              <Banner kind="ok">All benchmark processes finished.</Banner>
            </div>
          )}
        </>
      )}

      {activeDesignerJobs.length > 0 && (
        <div style={{ marginTop: runGroup ? 24 : 0 }}>
          <div className="spread" style={{ marginBottom: 12 }}>
            <h2 style={{ margin: 0 }}>Live — Designer Agent</h2>
            <div className="row">
              {designerSession.streamConnected && (
                <span className="badge completed">● streaming</span>
              )}
              {designerSession.running && designerSession.jobId && (
                <button
                  type="button"
                  className="danger"
                  onClick={(e) => {
                    e.stopPropagation();
                    cancelDesignerJob(designerSession.jobId!);
                  }}
                >
                  Stop
                </button>
              )}
            </div>
          </div>
          <div className="process-grid">
            {activeDesignerJobs.map((j) => (
              <div
                key={j.id}
                className="process-card"
                onClick={() => selectDesignerJob(j.id)}
              >
                <div className="pc-head">
                  <div>
                    <div className="pc-title">Designer Agent</div>
                    <div className="pc-sub mono small">{j.id}</div>
                  </div>
                  {statusBadge(j.status as Process["status"])}
                </div>
                <div className="small muted">
                  {(j.prompt || "").slice(0, 80) || "Space photo redesign"}
                </div>
                {j.upload_url && (
                  <img
                    className="live-designer-thumb"
                    src={j.upload_url}
                    alt="input"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (j.upload_url) openLightbox(j.upload_url);
                    }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ProcessCard({
  p,
  tcName,
  coName,
  onOpen,
}: {
  p: Process;
  tcName: string;
  coName: string;
  onOpen: () => void;
}) {
  const steps = p.steps;
  const completed = steps.filter((s) => s.status === "completed").length;
  const total = steps.length;
  const pct = total > 0 ? Math.min(100, (completed / total) * 100) : 0;
  const start = p.started_at ? new Date(p.started_at).getTime() : null;
  const elapsed = start ? Date.now() - start : 0;
  const cur = [...steps].reverse().find((s) => s.status === "running");

  return (
    <div className="process-card" onClick={onOpen}>
      <div className="pc-head">
        <div>
          <div className="pc-title">{coName}</div>
          <div className="pc-sub">{tcName}</div>
        </div>
        {statusBadge(p.status)}
      </div>
      <div className="small muted">
        {cur
          ? `step ${cur.index + 1}: ${cur.kind}${cur.model ? ` (${cur.model})` : ""}`
          : total > 0
            ? `step —`
            : "waiting…"}
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="spread small">
        <span className="muted">
          {completed}/{total || "?"} steps
        </span>
        <span className="mono">{fmtMs(elapsed)}</span>
      </div>
    </div>
  );
}
