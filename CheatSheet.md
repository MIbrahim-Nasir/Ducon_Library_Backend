# VPS Cheat Sheet

Active git branch for app work: **`main`** (FastprodCloudflare merged).

## Auth email domains

| Env | Values | Meaning |
|-----|--------|---------|
| `EMAIL_DOMAIN_POLICY` | `allowlist` (default) \| `allow_all` \| `block_disposable` | Server signup/login domain rules |
| `ALLOW_ALL_EMAIL_DOMAINS` | `true` / `false` | Convenience toggle → forces `allow_all` |
| Frontend | `VITE_EMAIL_DOMAIN_POLICY` / `VITE_ALLOW_ALL_EMAIL_DOMAINS` | Keep in sync with backend |

`block_disposable` uses the `disposable-email-domains` PyPI package (community temp-mail blocklist).

---

## Postgres

```bash
psql -U appducon_appuser -d appducon_duconlibrary04042026
```

App runtime uses `DATABASE_URL` (async: `postgresql+asyncpg://...`) from `.env` — see `VPS_DEPLOYMENT.md` / `env_template.txt`. Not the same as the interactive `psql` command above.

---

## systemd — `ducon-library`

Unit file: `/etc/systemd/system/ducon-library.service`  
Runs as user `appducon`, binds gunicorn to `127.0.0.1:8000`, loads `/home/appducon/Ducon_Library_Backend/.env`.

```bash
sudo systemctl start ducon-library
sudo systemctl restart ducon-library
sudo systemctl stop ducon-library
sudo systemctl reload ducon-library
sudo systemctl is-active ducon-library
```

After editing the unit file:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ducon-library
```

### Logs

```bash
sudo journalctl -u ducon-library -f
sudo journalctl -u ducon-library -n 100 --no-pager
```

---

## Add an admin

Privilege is `users.role` (`customer` | `admin` | `analytics`) — there is no `is_admin` flag.

### First admin (bootstrap via SQL)

User must already exist (signed up). Then promote:

```sql
UPDATE users SET role = 'admin' WHERE email = 'you@example.com';
-- read-only dashboards only:
UPDATE users SET role = 'analytics' WHERE email = 'analyst@example.com';
```

Also required for `/admin` access:

1. Set `ADMIN_PASSWORD_HASH` in `.env` — generate with `python -m scripts.hash_admin_password`
2. Restart the app (`sudo systemctl restart ducon-library`)

(`scripts/migrate_admin_otp.py` is a one-shot schema migration; its “next steps” print the same promote SQL.)

### Use the admin UI

1. Log in to the site as that user (normal JWT login).
2. Open `/admin` — enter the admin password → `POST /admin/auth/verify` issues a short-lived admin session (`X-Admin-Session`).
3. Full admin can change roles in the Users UI (`PATCH /admin/users/{id}` with `role`: `admin` / `analytics` / `customer`).

Roles:

| Role | Access |
|------|--------|
| `admin` | Settings, secrets, user roles, audit, metrics, errors |
| `analytics` | Read-only metrics / errors (no user CRUD / settings) |

---

## Apache proxy (HostGator / app.duconodl.com)

cPanel SSL **userdata** include (not `.htaccess`). Repo copies: `apache.userdata.ducon-proxy.conf.example`, `Local_Files/prod_configs/ducon-proxy.conf`.

**Path on VPS:**

```text
/etc/apache2/conf.d/userdata/ssl/2_4/appducon/app.duconodl.com/ducon-proxy.conf
```

`ProxyPass` is an **Apache config directive**, not a bash command — editing it in SSH as a shell line fails with `ProxyPass: command not found`. Put it in the file above, then rebuild/reload.

### Verify SSE / `/chat` (do not paste ProxyPass into bash)

```bash
grep -n 'ProxyPass.*chat' /etc/apache2/conf.d/userdata/ssl/2_4/appducon/app.duconodl.com/ducon-proxy.conf
grep -n flushpackets /etc/apache2/conf.d/userdata/ssl/2_4/appducon/app.duconodl.com/ducon-proxy.conf
```

Expect a line like:

```apache
ProxyPass        /chat                 http://127.0.0.1:8000/chat flushpackets=on
```

Without `flushpackets=on`, Apache buffers SSE and Cloudflare idle-kills (~100s).

### Edit → configtest → rebuild → restart

```bash
nano /etc/apache2/conf.d/userdata/ssl/2_4/appducon/app.duconodl.com/ducon-proxy.conf
# Do NOT wrap in <VirtualHost>. ProxyTimeout at top level only (not inside LocationMatch).

/scripts/ensure_vhost_includes --user=appducon   # once / if include is missing
apachectl configtest
/scripts/rebuildhttpdconf
/scripts/restartsrv_httpd
```

One-liner after the file is already included:

```bash
apachectl configtest && /scripts/rebuildhttpdconf && /scripts/restartsrv_httpd
```

(`VPS_DEPLOYMENT.md` also shows `systemctl reload apache2` / `apachectl graceful` for non-cPanel Apache; on this HostGator box use the `/scripts/*` sequence above.)
