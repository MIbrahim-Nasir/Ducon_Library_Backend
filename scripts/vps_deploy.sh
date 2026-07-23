#!/usr/bin/env bash
# Production deploy on the VPS (HostGator / appducon).
# Intended as ~/deploy.sh target or invoked by GitHub Actions after checkout sync.
#
# Steps: install deps into .venv → restart gunicorn → verify loopback /meta/build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> deploy: cwd=$ROOT commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

sync_deps() {
  if command -v uv >/dev/null 2>&1; then
    echo "==> deploy: uv sync"
    uv sync
  elif [[ -x .venv/bin/pip ]]; then
    echo "==> deploy: pip install . (pyproject deps into .venv)"
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install .
  else
    echo "ERROR: neither uv nor .venv/bin/pip found — cannot install deps" >&2
    exit 1
  fi
}

sync_deps

echo "==> deploy: restart ducon-library"
sudo systemctl restart ducon-library

echo "==> deploy: wait for gunicorn on 127.0.0.1:8000"
ok=0
for _ in $(seq 1 45); do
  if curl -sf --max-time 2 "http://127.0.0.1:8000/meta/build" >/dev/null; then
    ok=1
    break
  fi
  sleep 1
done

if [[ "$ok" -ne 1 ]]; then
  echo "ERROR: /meta/build did not become healthy within 45s" >&2
  sudo systemctl is-active ducon-library || true
  sudo journalctl -u ducon-library -n 120 --no-pager || true
  exit 1
fi

echo "==> deploy: healthy"
curl -sS --max-time 5 "http://127.0.0.1:8000/meta/build" || true
echo
