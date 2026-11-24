"""Ultimate release — rate limiting, single-worker guard, dashboard sign-in,
non-root container entrypoint, secret hygiene."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

import api.main as api_main  # noqa: E402
from api.main import RateLimiter, app, check_security_configuration  # noqa: E402
from config import settings as app_settings  # noqa: E402
from dashboard.components import auth  # noqa: E402


# --------------------------------------------------------------- rate limiting
class TestRateLimiter:
    def test_sliding_window(self):
        rl = RateLimiter()
        assert all(rl.hit(("1.2.3.4", "upload"), 3)[0] for _ in range(3))
        allowed, retry = rl.hit(("1.2.3.4", "upload"), 3)
        assert not allowed and retry >= 1
        assert rl.hit(("5.6.7.8", "upload"), 3)[0]          # other clients unaffected
        assert rl.hit(("1.2.3.4", "default"), 3)[0]         # other classes unaffected

    def test_tracked_clients_are_bounded(self):
        rl = RateLimiter()
        rl.MAX_TRACKED = 50
        for i in range(200):
            rl.hit((f"10.0.0.{i}", "default"), 5)
        assert len(rl._hits) == 50

    def test_429_with_retry_after_when_enabled(self):
        client = TestClient(app)
        api_main.rate_limiter.reset()
        with patch.object(app_settings, "RATE_LIMIT_ENABLED", True), \
             patch.object(app_settings, "RATE_LIMIT_COMPUTE_PER_MIN", 2):
            codes = [client.post("/ask", json={"question": "Which channel has the best ROI?"}).status_code
                     for _ in range(3)]
            last = client.post("/ask", json={"question": "Which channel has the best ROI?"})
        api_main.rate_limiter.reset()
        assert codes[:2] == [200, 200] and codes[2] == 429
        assert last.status_code == 429 and int(last.headers["Retry-After"]) >= 1
        assert last.json()["error"] == "rate_limited"

    def test_upload_has_its_own_tighter_limit_and_health_is_exempt(self):
        assert api_main._rate_class("POST", "/upload")[0] == "upload"
        assert api_main._rate_class("POST", "/datasets/abc/confirm")[0] == "compute"
        assert api_main._rate_class("GET", "/kpis")[0] == "default"
        assert api_main._rate_class("GET", "/health")[0] is None

    def test_disabled_in_development_by_default(self):
        client = TestClient(app)
        with patch.object(app_settings, "RATE_LIMIT_ENABLED", False):
            codes = {client.get("/channels").status_code for _ in range(30)}
        assert codes == {200}

    @pytest.mark.parametrize("env,expected", [
        ({}, "False"),
        ({"MARKETPULSE_ENV": "production"}, "True"),
        ({"MARKETPULSE_ENV": "production", "MARKETPULSE_RATE_LIMIT": "off"}, "False"),
        ({"MARKETPULSE_RATE_LIMIT": "on"}, "True"),
    ])
    def test_default_follows_environment(self, env, expected):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("MARKETPULSE_")}
        out = subprocess.run([sys.executable, "-c", "from config import settings; print(settings.RATE_LIMIT_ENABLED)"],
                             cwd=ROOT, env={**clean, **env}, capture_output=True, text=True, timeout=60)
        assert out.stdout.strip() == expected, out.stderr


class TestSingleWorker:
    def test_production_refuses_multiple_workers(self):
        with patch.object(app_settings, "APP_ENV", "production"), patch.object(app_settings, "API_KEY", "k"), \
             patch.dict(os.environ, {"WEB_CONCURRENCY": "4"}):
            with pytest.raises(RuntimeError, match="exactly one worker"):
                check_security_configuration()

    def test_development_only_warns(self):
        with patch.object(app_settings, "APP_ENV", "development"), patch.dict(os.environ, {"WEB_CONCURRENCY": "2"}):
            check_security_configuration()

    def test_docker_api_runs_one_worker(self):
        assert '"--workers", "1"' in (ROOT / "Dockerfile.api").read_text(encoding="utf-8")


# --------------------------------------------------------------- dashboard auth
class TestPasswordHashing:
    def test_round_trip_and_wrong_password(self):
        h = auth.hash_password("correct horse battery", iterations=1000)
        assert h.startswith("pbkdf2_sha256$1000$")
        assert auth.verify_password("correct horse battery", h)
        assert not auth.verify_password("correct horse batterY", h)
        assert not auth.verify_password("", h)

    def test_salted(self):
        assert auth.hash_password("x" * 12, iterations=1000) != auth.hash_password("x" * 12, iterations=1000)

    @pytest.mark.parametrize("bad", ["", "plain", "md5$1$a$b", "pbkdf2_sha256$notint$a$b", "pbkdf2_sha256$1$@@$@@"])
    def test_malformed_hash_never_authenticates(self, bad):
        assert not auth.verify_password("anything", bad)

    def test_no_plaintext_password_setting_exists(self):
        src = (ROOT / "dashboard" / "components" / "auth.py").read_text(encoding="utf-8") + (ROOT / ".env.example").read_text(encoding="utf-8")
        assert "DASHBOARD_PASSWORD=" not in src.replace("DASHBOARD_PASSWORD_HASH=", "")


class TestDashboardGate:
    def _app(self, env):
        from test_phase8_part1_foundation import _new_app, _demo_session
        with patch.dict(os.environ, env):
            at = _new_app(_demo_session())
            at.run()
        return at

    def test_development_without_hash_is_open(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(auth.ENV_NAME, None)
            at = self._app({})
        assert len(at.sidebar.radio) == 1

    def test_production_without_hash_refuses(self):
        os.environ.pop(auth.ENV_NAME, None)
        with patch.object(app_settings, "APP_ENV", "production"):
            at = self._app({})
        assert len(at.sidebar.radio) == 0
        assert any("not configured" in e.value for e in at.error)

    def test_hash_set_requires_sign_in_and_accepts_correct_password(self):
        from test_phase8_part1_foundation import _new_app, _demo_session
        h = auth.hash_password("team-password-123", iterations=1000)
        with patch.dict(os.environ, {auth.ENV_NAME: h}):
            at = _new_app(_demo_session())
            at.run()
            assert len(at.sidebar.radio) == 0                       # nothing rendered before sign-in
            at.text_input[0].input("wrong")
            at.button[0].click().run()
            assert any("Incorrect" in e.value for e in at.error)
            at.text_input[0].input("team-password-123")
            at.button[0].click().run()
            assert len(at.sidebar.radio) == 1                       # app visible after sign-in

    def test_lockout_after_repeated_failures(self):
        from test_phase8_part1_foundation import _new_app, _demo_session
        h = auth.hash_password("team-password-123", iterations=1000)
        with patch.dict(os.environ, {auth.ENV_NAME: h}):
            at = _new_app(_demo_session())
            at.run()
            for _ in range(auth.MAX_ATTEMPTS):
                at.text_input[0].input("nope")
                at.button[0].click().run()
            assert any("locked" in e.value.lower() or "too many" in e.value.lower() for e in at.error)
            at.run()
            assert len(at.text_input) == 0                          # form hidden while locked


# --------------------------------------------------------------- non-root entrypoint
@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0,
                    reason="privilege drop can only be exercised as root on Linux (the entrypoint runs only in Linux containers)")
class TestEntrypointPrivilegeDrop:
    @pytest.fixture
    def tmp_path(self):
        # pytest's own tmp root is 0700 (root-only), which an unprivileged uid
        # cannot traverse; a container's /app is world-readable, so mirror that.
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="mp_entry_", dir="/tmp"))
        os.chmod(d, 0o755)
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def _workdir(self, tmp_path, owner_uid=None):
        wd = tmp_path / "app"
        (wd / "data" / "processed").mkdir(parents=True)
        (wd / "models").mkdir()
        (wd / "data" / "processed" / "marketing_data_clean.csv").write_text("x\n")
        (wd / "models" / "mmm_model.pkl").write_bytes(b"m")
        shutil.copy(ROOT / "docker-entrypoint.sh", wd / "docker-entrypoint.sh")
        os.chmod(tmp_path, 0o755)
        os.chmod(wd, 0o755)
        if owner_uid is not None:
            for p in [wd / "data", *wd.joinpath("data").rglob("*"), wd / "models", *wd.joinpath("models").rglob("*")]:
                os.chown(p, owner_uid, owner_uid)
        return wd

    def _run(self, wd, *cmd, env=None):
        e = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": "/tmp", **(env or {})}
        return subprocess.run(["bash", "docker-entrypoint.sh", *cmd], cwd=wd, env=e,
                               capture_output=True, text=True, encoding="utf-8", timeout=60)

    def test_root_owned_volume_is_handed_to_app_user(self, tmp_path):
        wd = self._workdir(tmp_path)
        r = self._run(wd, "id", "-u")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == "10001"
        assert (wd / "data").stat().st_uid == 10001 and (wd / "models").stat().st_uid == 10001
        assert "skipping initialization" in r.stdout

    def test_host_owned_bind_mount_keeps_host_ownership(self, tmp_path):
        wd = self._workdir(tmp_path, owner_uid=1234)
        r = self._run(wd, "id", "-u")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == "1234"
        assert (wd / "data").stat().st_uid == 1234                  # not chowned to the container user
        assert (wd / "data" / "uploads").stat().st_uid == 1234

    def test_opt_out_runs_as_root(self, tmp_path):
        wd = self._workdir(tmp_path)
        r = self._run(wd, "id", "-u", env={"MARKETPULSE_RUN_AS_ROOT": "1"})
        assert r.stdout.strip().splitlines()[-1] == "0"

    def test_entrypoint_keeps_required_structure(self):
        s = (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")
        assert s.index("setuid") < s.index("flock -x 200")          # privileges dropped before init/app code
        assert 'exec "$@"' in s and "set -e" in s


# --------------------------------------------------------------- hygiene
class TestSecretHygiene:
    def test_env_example_has_no_values_for_secrets(self):
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if re.match(r"^(MARKETPULSE_API_KEY|API_KEY|ANTHROPIC_API_KEY|MARKETPULSE_DASHBOARD_PASSWORD_HASH)=", line):
                assert line.split("=", 1)[1].strip() == "", line

    def test_secrets_excluded_from_images(self):
        ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert ".env" in ignore and ".streamlit/secrets.toml" in ignore

    def test_reverse_proxy_configs_present(self):
        for f in ("Caddyfile", "nginx.conf", "docker-compose.proxy.yml"):
            assert (ROOT / "deploy" / f).exists()
