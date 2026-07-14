/**
 * flushStaleCache.ts
 * ──────────────────
 * Call once on initial page load (before rendering the app) so users never
 * keep a stale PWA/service-worker shell after a deploy.
 *
 * Flow:
 *  1. Fetch `/meta/build` with `cache: "no-store"` (backend must expose this).
 *  2. Compare `build_id` to the last seen value in session/local storage.
 *  3. On mismatch: delete Cache Storage entries, unregister service workers,
 *     persist the new build id, and hard-reload once.
 *
 * Usage (frontend bootstrap — before ReactDOM.createRoot):
 *
 *   import { flushStaleCache } from "@/utils/flushStaleCache";
 *   await flushStaleCache();
 *
 * Deploy: set `APP_BUILD_ID` on the backend to match the SPA build id
 * (git SHA or CI build number). The frontend can also pass `buildId` from
 * `import.meta.env.VITE_APP_BUILD_ID` when the meta endpoint is unreachable.
 */

const STORAGE_KEY = "ducon_app_build_id";
const RELOAD_FLAG = "ducon_cache_flush_reloaded";

export type FlushStaleCacheOptions = {
  /** Backend meta URL. Default: `/meta/build`. */
  buildUrl?: string;
  /** Optional build id from the SPA bundle (e.g. Vite env). */
  buildId?: string | null;
  /** Hard-reload after clearing caches when the build id changed. Default: true. */
  reload?: boolean;
};

export type FlushStaleCacheResult = {
  flushed: boolean;
  buildId: string | null;
};

export async function flushStaleCache(
  options: FlushStaleCacheOptions = {},
): Promise<FlushStaleCacheResult> {
  const buildUrl = options.buildUrl ?? "/meta/build";
  let buildId = options.buildId ?? null;

  if (!buildId) {
    buildId = await _fetchBuildId(buildUrl);
  }

  if (!buildId) {
    _clearReloadFlag();
    return { flushed: false, buildId: null };
  }

  const stored =
    sessionStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(STORAGE_KEY);

  if (stored && stored !== buildId) {
    await _clearBrowserCaches();
    _persistBuildId(buildId);

    if (options.reload !== false && !sessionStorage.getItem(RELOAD_FLAG)) {
      sessionStorage.setItem(RELOAD_FLAG, "1");
      window.location.reload();
      return { flushed: true, buildId };
    }
  } else if (!stored) {
    _persistBuildId(buildId);
  }

  _clearReloadFlag();
  return { flushed: false, buildId };
}

async function _fetchBuildId(buildUrl: string): Promise<string | null> {
  try {
    const res = await fetch(buildUrl, { cache: "no-store" });
    if (!res.ok) return null;
    const data = (await res.json()) as { build_id?: string; cache_version?: string };
    return data.build_id ?? data.cache_version ?? null;
  } catch {
    return null;
  }
}

async function _clearBrowserCaches(): Promise<void> {
  if ("caches" in window) {
    const names = await caches.keys();
    await Promise.all(names.map((name) => caches.delete(name)));
  }
  if ("serviceWorker" in navigator) {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map((reg) => reg.unregister()));
  }
}

function _persistBuildId(buildId: string): void {
  sessionStorage.setItem(STORAGE_KEY, buildId);
  localStorage.setItem(STORAGE_KEY, buildId);
}

function _clearReloadFlag(): void {
  sessionStorage.removeItem(RELOAD_FLAG);
}
