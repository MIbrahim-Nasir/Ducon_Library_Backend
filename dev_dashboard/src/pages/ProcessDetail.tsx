import { useMemo, useState } from "react";
import { useStore, dedupeSteps } from "../store";
import { Modal, fmtCost, fmtMs, statusBadge } from "../components/ui";
import type { Process, Step, TestCase } from "../types";

export default function ProcessDetail() {
  const selectedProcessId = useStore((s) => s.selectedProcessId);
  const selectProcess = useStore((s) => s.selectProcess);
  const processes = useStore((s) => s.processes);
  const testCases = useStore((s) => s.testCases);
  const combos = useStore((s) => s.combos);
  const setTab = useStore((s) => s.setTab);

  const procList = useMemo(() => Object.values(processes), [processes]);
  const tcById = useMemo(() => new Map(testCases.map((t) => [t.id, t])), [testCases]);
  const coById = useMemo(() => new Map(combos.map((c) => [c.id, c])), [combos]);

  if (!selectedProcessId) return null;
  const p = processes[selectedProcessId];
  if (!p) {
    return (
      <Modal title="Process" onClose={() => selectProcess(null)}>
        <div className="empty">Process not found.</div>
      </Modal>
    );
  }
  const idx = procList.findIndex((x) => x.id === selectedProcessId);
  const prev = idx > 0 ? procList[idx - 1] : null;
  const next = idx >= 0 && idx < procList.length - 1 ? procList[idx + 1] : null;

  return (
    <Modal
      title={
        <span>
          Process · {coById.get(p.combo_id)?.name ?? p.combo_id} ×{" "}
          {tcById.get(p.test_case_id)?.name ?? p.test_case_id}
        </span>
      }
      onClose={() => selectProcess(null)}
      footer={
        <div className="spread">
          <div className="row">
            <button disabled={!prev} onClick={() => prev && selectProcess(prev.id)}>
              ← Prev
            </button>
            <span className="small muted">
              {idx + 1} / {procList.length}
            </span>
            <button disabled={!next} onClick={() => next && selectProcess(next.id)}>
              Next →
            </button>
          </div>
          <div className="row">
            <button className="ghost" onClick={() => setTab("live")}>Back to Live</button>
            <button onClick={() => selectProcess(null)}>Close</button>
          </div>
        </div>
      }
    >
      <ProcessDetailBody p={p} tc={tcById.get(p.test_case_id)} />
    </Modal>
  );
}

function ProcessDetailBody({ p, tc }: { p: Process; tc?: TestCase }) {
  const displaySteps = useMemo(() => dedupeSteps(p.steps), [p.steps]);
  const inputImages = useMemo(() => {
    if (p.result?.input_images?.length) return p.result.input_images;
    if (!tc) return [];
    return [
      ...(tc.user_images ?? []).map((im) => ({
        url: im.url,
        role: "user" as const,
        metadata: im.metadata,
      })),
      ...(tc.ducon_images ?? []).map((im) => ({
        url: im.url,
        role: "ducon" as const,
        metadata: im.metadata,
      })),
    ].filter((im) => im.url);
  }, [p.result?.input_images, tc]);

  return (
    <div>
      <div className="spread" style={{ marginBottom: 12 }}>
        {statusBadge(p.status)}
        <div className="small muted">
          total {fmtMs(p.result?.total_duration_ms ?? elapsedMs(p))} ·{" "}
          cost {fmtCost(p.result?.total_cost_usd ?? sumCost(p))} ·{" "}
          retries {p.result?.retries ?? 0}
        </div>
      </div>

      {/* Input images */}
      <h3>Input images</h3>
      {inputImages.length === 0 ? (
        <div className="muted small" style={{ marginBottom: 12 }}>
          (no input images reported yet)
        </div>
      ) : (
        <div className="image-strip" style={{ marginBottom: 12 }}>
          {inputImages.map((im, i) => (
            <div className="image-tile" key={i} style={{ width: 96 }}>
              <img src={im.url} alt="" style={{ width: 96, height: 96 }} />
              <div className="tag">{im.role}</div>
              {im.metadata && (
                <div className="tag">{Object.values(im.metadata).slice(0, 2).join(" · ")}</div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Steps */}
      <h3>Steps ({displaySteps.length})</h3>
      <div className="step-list" style={{ marginBottom: 12 }}>
        {displaySteps.length === 0 && (
          <div className="muted small">No steps yet.</div>
        )}
        {displaySteps.map((s) => (
          <StepRow key={`${s.index}:${s.kind}`} s={s} />
        ))}
      </div>

      {/* Output images */}
      <h3>Output</h3>
      {(p.result?.output_images?.length ?? 0) === 0 ? (
        <div className="muted small" style={{ marginBottom: 12 }}>
          (no output yet)
        </div>
      ) : (
        <div className="image-strip" style={{ marginBottom: 12 }}>
          {p.result?.output_images?.map((im, i) => (
            <a key={i} href={im.url} target="_blank" rel="noreferrer">
              <img className="thumb lg" src={im.url} alt={`output ${i}`} />
            </a>
          ))}
        </div>
      )}

      {/* Cost breakdown */}
      <h3>Cost breakdown</h3>
      {p.result?.cost_breakdown && p.result.cost_breakdown.length > 0 ? (
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Tokens</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {p.result.cost_breakdown.map((row, i) => (
              <tr key={i}>
                <td className="mono small">{row.model}</td>
                <td className="mono small">{row.tokens}</td>
                <td className="mono small">{fmtCost(row.cost_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="muted small">No cost breakdown available.</div>
      )}

      {p.result?.final_prompt && (
        <div style={{ marginTop: 12 }}>
          <h3>Final prompt</h3>
          <pre className="prompt">{p.result.final_prompt}</pre>
        </div>
      )}
    </div>
  );
}

function StepRow({ s }: { s: Step }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="step">
      <div className="step-head">
        <div className="row">
          <span className="tag-pill">{s.kind}</span>
          {s.model && <span className="mono small">{s.model}</span>}
          {s.thinking && <span className="tag-pill">{s.thinking}</span>}
        </div>
        <div className="row small">
          {statusBadge(s.status)}
          <span className="mono muted">{fmtMs(s.duration_ms)}</span>
          {s.prompt_used && (
            <button className="ghost" onClick={() => setOpen((v) => !v)}>
              {open ? "hide prompt" : "show prompt"}
            </button>
          )}
        </div>
      </div>
      <div className="small">
        <span className="muted">tokens in:</span> {s.tokens_in ?? "—"}{" "}
        <span className="muted">out:</span> {s.tokens_out ?? "—"}{" "}
        <span className="muted">cost:</span> {fmtCost(s.cost_usd)}
        {s.error && <div className="badge failed" style={{ marginTop: 6 }}>{s.error}</div>}
      </div>
      {open && s.prompt_used && (
        <div className="step-body">
          <pre className="prompt">{s.prompt_used}</pre>
        </div>
      )}
    </div>
  );
}

function elapsedMs(p: Process): number | undefined {
  if (!p.started_at) return undefined;
  const end = p.ended_at ? new Date(p.ended_at).getTime() : Date.now();
  return end - new Date(p.started_at).getTime();
}
function sumCost(p: Process): number | undefined {
  const stepTotal = p.steps.reduce((a, s) => a + (s.cost_usd ?? 0), 0);
  return stepTotal || undefined;
}
