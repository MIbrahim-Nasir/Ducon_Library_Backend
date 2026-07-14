# Ducon Dev Dashboard — INTERNAL DEV TOOL

> ⚠️ **INTERNAL DEV TOOL — DO NOT DEPLOY TO PRODUCTION.**
>
> This is a standalone Vite + React + TypeScript app used for **developer
> experimentation only**: running combinations of AI generation workflows
> against saved test cases and inspecting every step with full metrics. It is
> **completely separate** from the main production frontend (which lives in a
> different repo at `Ducon_Library`) and is **never** imported by the FastAPI
> backend or the main frontend.
>
> This app is intentionally **not** wired into any production build or deploy
> pipeline. See the note appended to `../VPS_DEPLOYMENT.md`.

## Run it

From this folder:

```bash
npm install
npm run dev
```

The dev server starts on **http://localhost:5175** (Vite default, strict port).

It expects the local FastAPI backend to be running at
**http://localhost:8000** with the `/dev/*` endpoints implemented. The Vite
config proxies `/dev` → `http://localhost:8000` (see `vite.config.ts`).

If the backend is **not** running, the dashboard still loads — it shows a
"backend offline" banner and falls back to localStorage for test cases and
combos. Starting a run without the backend will fabricate a local-only run
group with no real generation (so the UI flow can be exercised).

### Build

```bash
npm run build      # type-checks (tsc -b) then vite build → dist/
npm run preview    # preview the production build on :5175
```

`npm run build` must pass before this is considered done.

## Layout

- `src/api.ts` — single typed API client for every `/dev` endpoint + SSE helper.
- `src/store.ts` — zustand store; the single source of UI state. Test cases
  and combos are persisted to `localStorage` and merged with the backend.
- `src/types.ts` — shared types mirroring the `/dev` API contract.
- `src/components/` — `Sidebar` (nav + image catalog), shared `ui` helpers.
- `src/pages/` — `TestCases`, `Combos`, `RunQueue`, `Live`, `ProcessDetail`,
  `Results`, `History`.

## Tabs

1. **Test Cases** — saved input bundles (user uploads + ducon catalog images +
   `use_ducon_data` toggle + optional hint). Persisted to localStorage.
2. **Combinations** — named configs of flow / models / thinking modes / rounds.
3. **Run Queue** — pick N test cases × M combos (cartesian product), Start.
4. **Live** — all concurrent processes as cards with live SSE updates, step
   progress, elapsed timers. Click a card for the Process Detail overlay.
5. **Process Detail** — chronological step list (kind, model, thinking,
   duration, collapsible prompt, tokens, cost, status), input/output images,
   cost breakdown, totals. Prev/Next to move between processes.
6. **Results / Compare** — benchmark table (one row per process) with status,
   time, retries, cost, output thumbnail. Click a row → Process Detail.
7. **History** — past run groups from `/dev/history`; reopen one into Results.

The left side panel always shows the full image catalog (searchable, type
filter) alongside navigation.

## Notes

- SSE uses `EventSource` on `/dev/runs/{id}/stream`; a 5s snapshot poll is a
  fallback for reconnect gaps.
- Every control is functional — there are no placeholder buttons. Endpoints
  that aren't implemented yet degrade to a clear "backend not connected" state.
- Dark theme, plain CSS (`src/styles.css`), no UI libraries.
