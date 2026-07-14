# Ducon production Selenium E2E

UI smoke tests against the live Ducon app. They open modals, navigate studio steps, and log in — they do **not** run paid AI generation pipelines unless you explicitly opt in.

## Working base URL

| URL | Status (2026-07-11) |
|-----|---------------------|
| `https://app.duconodl.com` | **Works** (DNS → `50.6.6.104`, HTTP 200) — default |
| `https://app.ducon.com` | Fallback only if primary fails to load |

Override with `E2E_BASE_URL`.

## Install

```bash
pip install -r requirements-e2e.txt
```

Requires Google Chrome. Selenium 4.6+ uses Selenium Manager for ChromeDriver; `webdriver-manager` is a fallback.

## Run

E2E tests are **excluded** from a plain `pytest tests/` run. Select them explicitly:

```bash
# Headed (recommended first run — easier to debug Turnstile)
pytest -m e2e tests/e2e -v

# Headless
set E2E_HEADLESS=1
pytest -m e2e tests/e2e -v

# Custom credentials / URL
set E2E_BASE_URL=https://app.duconodl.com
set E2E_EMAIL=test@gmail.com
set E2E_PASSWORD=test@123
pytest -m e2e tests/e2e -v

# Single smoke
pytest -m e2e tests/e2e/test_guest.py::test_guest_lands_and_sees_hero -v
```

PowerShell:

```powershell
$env:E2E_HEADLESS="1"
pytest -m e2e tests/e2e -v
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `E2E_BASE_URL` | `https://app.duconodl.com` | App origin |
| `E2E_EMAIL` | `test@gmail.com` | Login email |
| `E2E_PASSWORD` | `test@123` | Login password |
| `E2E_HEADLESS` | `0` | `1` = headless Chrome |
| `E2E_ALLOW_GEN` | `0` | `1` = allow sending chat messages that may hit the API |
| `E2E_EXPLICIT_WAIT` | `25` | WebDriverWait seconds |

Do not commit real secrets; override via env in CI/local.

## Coverage

| Area | What the suite does |
|------|---------------------|
| Guest land | Consent wall, onboarding dismiss, hero visible |
| Catalog | Designs / Products / Areas tabs, browse scroll |
| Bookmarks | Click bookmark on a catalog card; open Bookmarks panel |
| Uploads | Open Uploads modal, upload `fixtures/sample_upload.png` (local IndexedDB) |
| AI Generations | Open panel; guest sees sign-in gate; auth sees list/empty UI |
| Chat | Open Designer chat, assert composer; send only if `E2E_ALLOW_GEN=1` |
| Studio | Open “Visualize my space”, assert wizard steps + upload UI — **no Visualize run** |
| Auth | Login, Saved/Logout chrome, authenticated variants of above, logout |
| Search | Open search control (auth test) |

## Intentionally skipped / gated

- **Paid AI generations** (studio visualize, multi-image gen, chat tool gens) — UI smoke only. Gate any gen-touching path behind `E2E_ALLOW_GEN=1`.
- **Cloudflare Turnstile** — if a challenge iframe / “Verify you are human” blocks the page, tests call `pytest.skip` with a clear message. Mitigations: headed browser + manual solve, allowlisted IP, or Turnstile test keys on a non-prod env.
- **Voice AI** — not automated (mic permissions).

## Layout

```
tests/e2e/
  conftest.py          # Chrome fixture, markers, guest/login fixtures
  config.py            # Env config
  helpers.py           # Waits, Turnstile detect, overlays
  pages/               # Page objects (home, auth, sidebar, chat, studio)
  fixtures/sample_upload.png
  test_guest.py
  test_auth.py
  README.md
```

## Notes for production

- Prefer smoke assertions over full gen pipelines.
- Uploads in the sidebar are stored in **browser IndexedDB** (local), so fixture uploads are cheap.
- Guest consent is stored in `localStorage` key `ducon_guest_consented`.
