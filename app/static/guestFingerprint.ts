/**
 * guestFingerprint.ts
 * ───────────────────
 * Collects a stable per-device identifier via FingerprintJS OSS, hashes it with
 * SHA-256, and returns a 64-char hex string for the X-Guest-Fingerprint header.
 *
 * Install in the frontend app:
 *   npm install @fingerprintjs/fingerprintjs
 *
 * Key properties:
 *  • Survives localStorage.clear(), sessionStorage.clear(), and cookie deletion.
 *  • Usually survives incognito / private mode on the same device.
 *  • Privacy-safe: only the hashed digest leaves the browser (never the raw visitorId).
 *
 * Limitations:
 *  • Privacy browsers (Tor, Brave strict, Firefox fingerprint-resistance) may
 *    produce a fresh visitorId each session — expected and acceptable.
 *  • Two identical machines with the same OS + browser build can share a fingerprint
 *    (rare; they get the same quota which is still fairer than sharing by IP).
 *
 * Usage — call once early in your app and cache the result:
 *
 *   import { getGuestFingerprint } from "@/lib/guestFingerprint";
 *
 *   const fp = await getGuestFingerprint();
 *   headers["X-Guest-Fingerprint"] = fp;
 */

import FingerprintJS from "@fingerprintjs/fingerprintjs";

// ── Cached promise so we only compute once per page load ─────────────────────
let _cached: Promise<string> | null = null;

export function getGuestFingerprint(): Promise<string> {
  if (!_cached) {
    _cached = _computeFingerprint().catch(() => "");
  }
  return _cached;
}

// ── Internal computation ──────────────────────────────────────────────────────

async function _computeFingerprint(): Promise<string> {
  const agent = await FingerprintJS.load();
  const result = await agent.get();
  return _sha256(result.visitorId);
}

// ── SHA-256 via Web Crypto (available in all modern browsers) ─────────────────

async function _sha256(input: string): Promise<string> {
  const encoded = new TextEncoder().encode(input);
  const hashBuffer = await crypto.subtle.digest("SHA-256", encoded);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}
