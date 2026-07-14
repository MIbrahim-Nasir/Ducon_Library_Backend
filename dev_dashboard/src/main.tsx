import React from "react";
import ReactDOM from "react-dom/client";
import { flushStaleCache } from "./lib/flushStaleCache";
import App from "./App";
import "./styles.css";

async function bootstrap() {
  await flushStaleCache();
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
}

void bootstrap();
