# Deploying MarketPulse AI

This folder holds the HTTPS reverse-proxy configuration and the deployment steps.
What was actually executed is recorded in `../FINAL_DEPLOYMENT_VERIFICATION.md`.
Anything not listed there has not been verified.

## 1. Prepare secrets (all deployments)

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # -> MARKETPULSE_API_KEY
python -m dashboard.components.auth                            # prompts for a password, prints its hash
```

In `.env`:

```
MARKETPULSE_ENV=production
MARKETPULSE_API_KEY=<the token>
MARKETPULSE_DASHBOARD_PASSWORD_HASH=pbkdf2_sha256$$600000$$<salt>$$<hash>
CORS_ORIGINS=https://dashboard.example.com
```

**Escape the hash.** Docker Compose interpolates `$` in `.env`, so every `$` in
the generated hash must be written as `$$`. If you set the variable directly in
the shell environment or in Streamlit secrets, use the hash exactly as printed,
with single `$`.

In production:

- the API refuses to start without `MARKETPULSE_API_KEY`;
- the dashboard refuses to serve pages without `MARKETPULSE_DASHBOARD_PASSWORD_HASH`;
- API rate limiting is on (10 uploads, 30 compute calls and 120 other requests per
  minute per client IP; configurable).

## 2. Docker Compose (official base image)

```bash
docker compose build
docker compose up -d
docker compose ps                     # both services should become "healthy"
curl -fsS http://localhost:8000/health
```

- The dashboard is on port 8501 and the API on port 8000.
- `./data` and `./models` are bind-mounted.
- On first start the entrypoint generates the demo artifacts, but only where they are missing.
- The entrypoint then drops root. The app runs as the owner of `./data` when that
  is a non-root host user; otherwise it runs as uid/gid 10001, and `./data` and
  `./models` are chowned to that user.
- `MARKETPULSE_RUN_AS_ROOT=1` disables the privilege drop. It is not recommended.
- The API runs with exactly one worker. Do not scale it with `--workers` or
  `WEB_CONCURRENCY`: uploads and rate limits are tracked in memory per process.

> **Verification note:** in the release environment Docker Hub was unreachable,
> so these images were built and run against a locally constructed substitute
> tagged `python:3.13-slim`. A build against the real official image was **not**
> performed. Run the commands above on a machine with registry access to confirm.

## 3. HTTPS with Caddy (recommended)

1. Point DNS for your two hostnames at the server. Open ports 80 and 443.
2. Edit `deploy/Caddyfile`:
   - replace `dashboard.example.com` and `api.example.com` with your hostnames;
   - replace `ops@example.com` with your contact email.
3. Start everything with the overlay:

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.proxy.yml up -d
```

The overlay:

- stops publishing ports 8501 and 8000, so only Caddy is reachable;
- adds Caddy with automatic Let's Encrypt certificates and security headers (HSTS,
  nosniff, frame denial);
- sets request body limits;
- sets `FORWARDED_ALLOW_IPS="*"` for the API, so uvicorn trusts the proxy's
  `X-Forwarded-For` and rate limits apply per real client IP.

`FORWARDED_ALLOW_IPS="*"` is only safe because the API port is not published. If
you publish 8000 again, remove that setting or restrict it to the proxy's address.
Otherwise clients could spoof their IP and evade the rate limit.

Requires Docker Compose v2.24 or later (for `!reset`).

## 4. HTTPS with nginx (alternative)

`deploy/nginx.conf` is an equivalent server configuration:

- TLS
- the WebSocket upgrade required by Streamlit
- `limit_req`
- forwarded headers

Install your certificates, for example with certbot, at the paths named in the
file. Proxy to the containers on localhost, and bind the published ports to
127.0.0.1 only.

## 5. Streamlit Community Cloud

1. Push the project to a GitHub repository you control.
2. In Streamlit Community Cloud, create an app:
   - main file: `app.py`;
   - Advanced settings: Python 3.13.
3. Under **Secrets**, add:

   ```toml
   MARKETPULSE_DASHBOARD_PASSWORD_HASH = "pbkdf2_sha256$600000$<salt>$<hash>"
   ```

   Use single `$` here; this is TOML, not Compose. Add `MARKETPULSE_ENV = "production"`
   if you want the app to refuse to serve when the hash is missing.
4. Deploy.

Notes:

- The dashboard does not call the FastAPI backend, so API keys, rate limits and
  CORS do not apply on Streamlit Cloud.
- Uploaded files live on the app's ephemeral disk and disappear when it restarts.
- Community Cloud resources are limited. The performance report measured a
  1,000,000-row analysis at about 1.8 GB peak memory, which may exceed the
  free tier.
- `ANTHROPIC_API_KEY` is not used by any code path.

This Streamlit Cloud procedure was written from the configuration in this
repository. It was **not** executed for this release.

## 6. Rollback

Keep the previous release folder and `./data` and `./models` backups. Then run:

```bash
docker compose down
# restore the previous release folder (and data/models backup if needed)
docker compose up -d --build
```
