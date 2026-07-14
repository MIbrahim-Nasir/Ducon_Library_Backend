# Deployment Endpoint Checklist

This checklist complements the automated deploy tests. The default runner keeps
external services mocked or avoided, so items marked manual/env-dependent should
be verified in staging or a local environment with the required credentials.

## Automated In `tests/`

- `POST /generate-multi-image`
  - Missing auth and guest session returns `401`.
  - Invalid guest UUID returns `400`.
  - Validation rejects missing `images_meta`, bad JSON, empty arrays, too many images, invalid file refs, and missing labels.
  - Logged-in requests build a stream without guest-session commit behavior.
  - Guest requests commit the guest session before the SSE worker starts.
  - Guest SSE worker re-loads the `GuestSession` in its own DB session before saving.
- `app.tool_generate_image.generate_multi_image`
  - Regression coverage confirms `_run_generation_sync` receives `image_thinking` and `metrics` as keyword arguments through `asyncio.to_thread`.
- `app.chat_agent.get_chat_tools`
  - Guests do not receive `start_designer_job`; authenticated users do.
- Frontend `Ducon_Library/src/App.jsx` (when sibling repo is present)
  - `start_designer_job` bridge `useEffect` dependency array includes `user`.

## Manual Or Env-Dependent Before Deploy

- `POST /generate-multi-image`
  - Staging happy path for authenticated users with catalog, generation, URL, and uploaded-file sources.
  - Staging happy path for guests with valid Turnstile token and guest rate-limit counters.
  - Storage verification for saved authenticated and guest generations, including signed URL access.
- Auth endpoints under `/auth`
  - Signup OTP request/verify, login, Google login, forgot/reset password, logout, profile update, delete account.
  - Token expiry/revocation behavior and `/auth/me`, `/auth/role`, consent endpoints.
- Guest endpoints under `/guest` and `/auth/claim-guest-generations`
  - Usage snapshots, generation image serving, claim flow, cleanup cron with secret.
- Chat and voice endpoints under `/chat` and `/voice`
  - Guest and logged-in message flows, tool-result continuation, voice context, browse/studio context.
  - Turnstile and rate-limit behavior for guest flows.
- Catalog/user endpoints
  - `/images/metadata`, `/bookmarks`, `/generations`, `/generations/{id}/image`.
- Designer/contact/quotation flows
  - `/designer/jobs`, `/contact/designer`, `/contact/customer-service`, `/quotation`.
- Admin/dev endpoints
  - Admin OTP/auth status, settings updates/reveals, metrics, user role changes, audit log, error log.
  - `/dev/*` benchmark endpoints only when dev access is enabled.
