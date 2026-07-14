import { useEffect, type ReactNode } from "react";
import { useStore } from "./store";
import Sidebar from "./components/Sidebar";
import TestCases from "./pages/TestCases";
import Combos from "./pages/Combos";
import DesignerAgent from "./pages/DesignerAgent";
import RunQueue from "./pages/RunQueue";
import Live from "./pages/Live";
import ProcessDetail from "./pages/ProcessDetail";
import Results from "./pages/Results";
import History from "./pages/History";
import Lightbox from "./components/Lightbox";

export default function App() {
  const activeTab = useStore((s) => s.activeTab);
  const ping = useStore((s) => s.ping);
  const refreshCatalog = useStore((s) => s.refreshCatalog);
  const refreshModels = useStore((s) => s.refreshModels);
  const refreshTestCases = useStore((s) => s.refreshTestCases);
  const refreshCombos = useStore((s) => s.refreshCombos);
  const refreshHistory = useStore((s) => s.refreshHistory);
  const reconnectDesignerStream = useStore((s) => s.reconnectDesignerStream);
  const designerSession = useStore((s) => s.designerSession);

  useEffect(() => {
    ping();
    refreshCatalog();
    refreshModels();
    refreshTestCases();
    refreshCombos();
    refreshHistory();
    const id = setInterval(() => ping(), 30000);
    return () => clearInterval(id);
  }, [ping, refreshCatalog, refreshModels, refreshTestCases, refreshCombos, refreshHistory]);

  // Single source of truth for designer SSE reconnect: survives tab switches
  // (all tabs stay mounted) and re-opens the stream whenever the active job or
  // its running flag changes. The store guards against reopening the same job.
  useEffect(() => {
    if (designerSession.jobId && designerSession.running) {
      reconnectDesignerStream(designerSession.jobId);
    }
  }, [designerSession.jobId, designerSession.running, reconnectDesignerStream]);

  const pane = (tab: typeof activeTab, node: ReactNode) => (
    <div className={activeTab === tab ? "tab-pane" : "tab-pane hidden"}>{node}</div>
  );

  return (
    <div className="app">
      <Sidebar />
      <main className="main">
        {pane("test-cases", <TestCases />)}
        {pane("combos", <Combos />)}
        {pane("designer-agent", <DesignerAgent />)}
        {pane("run-queue", <RunQueue />)}
        {pane("live", <Live />)}
        {pane("results", <Results />)}
        {pane("history", <History />)}
      </main>
      <ProcessDetail />
      <Lightbox />
    </div>
  );
}
