import { useMemo } from "react";
import { useStore } from "../store";
import { Banner, Empty } from "../components/ui";

export default function RunQueue() {
  const testCases = useStore((s) => s.testCases);
  const combos = useStore((s) => s.combos);
  const selectedTc = useStore((s) => s.selectedTestCaseIds);
  const selectedCo = useStore((s) => s.selectedComboIds);
  const toggleTc = useStore((s) => s.toggleTestCaseSelected);
  const toggleCo = useStore((s) => s.toggleComboSelected);
  const selectAllTc = useStore((s) => s.selectAllTestCases);
  const selectAllCo = useStore((s) => s.selectAllCombos);
  const startRun = useStore((s) => s.startRun);
  const backendUp = useStore((s) => s.backendUp);

  const tcList = useMemo(() => Array.from(selectedTc), [selectedTc]);
  const coList = useMemo(() => Array.from(selectedCo), [selectedCo]);
  const productCount = tcList.length * coList.length;

  return (
    <div>
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Run Queue</h2>
        <div className="row">
          <span className="muted small">
            {tcList.length} test cases × {coList.length} combos ={" "}
            <strong>{productCount}</strong> processes
          </span>
          <button
            className="primary"
            disabled={productCount === 0}
            onClick={() => startRun()}
          >
            ▶ Start
          </button>
        </div>
      </div>

      {!backendUp && (
        <Banner kind="err">
          Backend not connected — starting a run will fabricate a local-only run
          group with no real generation. Bring the backend up at
          <code> http://localhost:8000</code> to actually execute.
        </Banner>
      )}

      <div className="grid cols-2">
        <div className="card">
          <div className="spread" style={{ marginBottom: 8 }}>
            <h3 style={{ margin: 0 }}>Test Cases</h3>
            <div className="row">
              <button className="ghost small" onClick={() => selectAllTc(true)}>all</button>
              <button className="ghost small" onClick={() => selectAllTc(false)}>none</button>
            </div>
          </div>
          {testCases.length === 0 ? (
            <Empty>No test cases. Create some first.</Empty>
          ) : (
            <ul className="bare col">
              {testCases.map((tc) => (
                <li key={tc.id} className="row">
                  <input
                    type="checkbox"
                    className="checkbox"
                    checked={selectedTc.has(tc.id)}
                    onChange={() => toggleTc(tc.id)}
                  />
                  <span>
                    <strong>{tc.name}</strong>{" "}
                    <span className="muted small">
                      ({(tc.user_images ?? []).length}u · {(tc.ducon_images ?? []).length}d
                      {tc.use_ducon_data ? " · ducon" : ""})
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card">
          <div className="spread" style={{ marginBottom: 8 }}>
            <h3 style={{ margin: 0 }}>Combos</h3>
            <div className="row">
              <button className="ghost small" onClick={() => selectAllCo(true)}>all</button>
              <button className="ghost small" onClick={() => selectAllCo(false)}>none</button>
            </div>
          </div>
          {combos.length === 0 ? (
            <Empty>No combos. Create some first.</Empty>
          ) : (
            <ul className="bare col">
              {combos.map((c) => (
                <li key={c.id} className="row">
                  <input
                    type="checkbox"
                    className="checkbox"
                    checked={selectedCo.has(c.id)}
                    onChange={() => toggleCo(c.id)}
                  />
                  <span>
                    <strong>{c.name}</strong>{" "}
                    <span className="muted small">
                      ({c.flow} · {c.image_model})
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
