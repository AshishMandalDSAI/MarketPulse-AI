# MarketPulse AI: Known Limitations

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22
**Supersedes:** the known-limitations list of MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip.

No Critical or High issue identified during this release's work remains open. The items below are accepted limitations, verification gaps caused by the environment, or decisions left to the operator.

## Resolved since the previous release

These items were on the previous list and are now fixed:

| Previous limitation | Status |
|---|---|
| No API rate limiting | **Resolved.** An in-process sliding-window limiter returns HTTP 429 with `Retry-After`. It is on by default in production and configurable. |
| Containers run as root | **Resolved.** The entrypoint drops to uid/gid 10001, or to the owner of `./data` if that is a non-root user. Verified live (substitute base image, see below). |
| Dashboard had no authentication | **Resolved.** Optional single team password stored as a PBKDF2-SHA256 hash (600,000 iterations), with lockout after 5 failures. It is **mandatory in production**. |
| No HTTPS guidance | **Resolved.** `deploy/` contains Caddy (automatic HTTPS) and nginx configurations plus a Compose overlay. |
| No dependency vulnerability scan | **Resolved.** pip-audit was run; see `FINAL_DEPENDENCY_SECURITY_REPORT.md`. |
| Multi-worker API could lose uploads | **Mitigated.** The API is pinned to one worker; production refuses `WEB_CONCURRENCY > 1`. |

## Analytical

1. **Associations, not causation.** MMM coefficients, ROI/ROAS, driver importances and correlations describe statistical relationships, not causal effects.
2. **Estimates, not guarantees.** Forecasts and optimised budgets are model estimates. Intervals are approximate (±1.645 × out-of-sample error SD).
3. **Modest demo model fit.** The demo MMM has test R² 0.35 and adjusted R² 0.045 on 42 held-out weeks. See `FINAL_ANALYTICS_ACCURACY_REPORT.md`.
4. **Synthetic data only.** The demo dataset is synthetic (209 weekly rows), and every accuracy figure in the reports comes from synthetic data. Accuracy on real business data is not verified.
5. **Heuristic column roles.** HIGH-confidence roles are applied automatically. MEDIUM- and LOW-confidence guesses are used only after the user confirms them.
6. **Ambiguous date order.** When no value in a `dd/mm` vs `mm/dd` column has a day above 12, the column is read month-first and the ambiguity is reported in the processing log. The user must check the date range.
7. **Sampling on large data.** Large inputs are sampled in some stages:
   - anomaly detection, driver analysis and the MMM engine use at most 50,000 rows;
   - the prediction engines use at most 100,000 rows;
   - segmentation chooses k on a 5,000-row sample.

   Each stage says so in its output.
8. **The AI Marketing Analyst is rule-based.** It routes questions by keyword and answers from computed results. It is **not a language model**, and no code path calls an LLM. `ANTHROPIC_API_KEY` is unused, and the `anthropic` package was removed from `requirements.txt` because nothing imported it. Decision Intelligence "copilot" routing is also keyword-based.
9. **The What-If simulator uses the demo model only.** For uploaded data, the analyst points to Prescriptive Analytics (module 17) instead of reusing the demo simulator.

## Security and operations

1. **One shared dashboard password.** There are no per-user accounts, roles or audit log of who did what. Use a proxy with SSO if you need identities.
2. **In-process rate limiter.** The rate limiter and the upload registry live in a single process. Limits reset on restart, and horizontal scaling needs a shared store (for example Redis), which is not implemented.
3. **Rate limits apply to the API only.** The Streamlit dashboard has no request rate limiting beyond the proxy's (`deploy/nginx.conf` includes `limit_req`).
4. **Single app-level TLS path.** TLS is terminated by the proxy. The app itself serves plain HTTP inside the container network.
5. **`$` escaping in Compose.** `docker compose` interpolates `$` in `.env`, so the password hash must be written with `$$`. This is documented in `.env.example` and `deploy/README.md`.

## Verification gaps (environment)

1. **Official Docker base image not tested.** Docker Hub was unreachable, so the Docker build and run were verified against a locally built substitute tagged `python:3.13-slim`. That substitute was made from this host's Python 3.13.18 and has no apt. The exact local commands to repeat the check on the official image are in `deploy/README.md` §2.
2. **Not deployed anywhere.** No HTTPS proxy with a real certificate, no Streamlit Community Cloud deployment and no public deployment were performed. The hosted link in README.md runs an earlier release.
3. **pip-audit used the PyPI vulnerability service only.** The OSV service was blocked by the network allowlist.
4. **Only Chrome was tested.** Browser testing used Chrome (Playwright) at 1440, 820 and 390 px widths. Firefox and Safari were not tested.
5. **No load testing.** Concurrent-user load was not tested.

## Input formats

1. CSV and `.xlsx` only. Legacy `.xls` is rejected with guidance.
2. One header row and one record per row. Multi-row headers and pivoted layouts are not detected.
3. The placeholder vocabulary for missing values is English (`n/a`, `null`, `none`, `missing`, …).
