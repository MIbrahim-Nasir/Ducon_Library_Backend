import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// INTERNAL DEV TOOL — DO NOT DEPLOY TO PRODUCTION
// Dev-only Vite config. The /dev proxy points at the local FastAPI backend.
// strictPort is false so a stale instance on 5175 doesn't block startup —
// vite falls back to the next free port and prints the actual URL.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    strictPort: false,
    proxy: {
      "/meta": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/dev": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: false,
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq, req) => {
            // Only override Accept for SSE stream routes — setting it on all
            // /dev requests breaks JSON POST/GET responses through the proxy.
            if (req.url?.includes("/stream")) {
              proxyReq.setHeader("Accept", "text/event-stream");
            }
          });
          proxy.on("proxyRes", (proxyRes, req) => {
            if (req.url?.includes("/stream")) {
              proxyRes.headers["cache-control"] = "no-cache";
              proxyRes.headers["x-accel-buffering"] = "no";
            }
          });
        },
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
