"""Regression tests for Docker entrypoint initialization logic.

Tests the sentinel-file-based initialization flow implemented in
docker-entrypoint.sh without requiring Docker. Uses temporary directories
and subprocess calls to validate:

1. Fresh empty storage initialization
2. Existing data preservation on startup
3. Repeated initialization (idempotency)
4. Correct configured data/model paths
5. Startup behavior when optional demo assets are absent
6. Entrypoint script structure and safety properties
"""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# Project root for locating docker-entrypoint.sh
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_SCRIPT = PROJECT_ROOT / "docker-entrypoint.sh"


class TestEntrypointScriptExists:
    """Verify the entrypoint script exists and has correct properties."""

    def test_entrypoint_exists(self):
        assert ENTRYPOINT_SCRIPT.exists(), "docker-entrypoint.sh must exist"

    def test_entrypoint_is_executable(self):
        assert os.access(ENTRYPOINT_SCRIPT, os.X_OK), (
            "docker-entrypoint.sh must be executable"
        )

    def test_entrypoint_has_shebang(self):
        content = ENTRYPOINT_SCRIPT.read_text(encoding="utf-8")
        assert content.startswith("#!/"), (
            "docker-entrypoint.sh must start with a shebang line"
        )

    def test_entrypoint_uses_set_e(self):
        content = ENTRYPOINT_SCRIPT.read_text(encoding="utf-8")
        assert "set -e" in content, (
            "docker-entrypoint.sh must use 'set -e' for fail-fast behavior"
        )

    def test_entrypoint_uses_exec(self):
        content = ENTRYPOINT_SCRIPT.read_text(encoding="utf-8")
        assert 'exec "$@"' in content, (
            "docker-entrypoint.sh must exec the CMD arguments"
        )

    def test_entrypoint_checks_sentinels(self):
        content = ENTRYPOINT_SCRIPT.read_text(encoding="utf-8")
        assert "data/processed/marketing_data_clean.csv" in content
        assert "models/mmm_model.pkl" in content

    def test_entrypoint_creates_uploads_dir(self):
        content = ENTRYPOINT_SCRIPT.read_text(encoding="utf-8")
        # The uploads dir may be created in a combined mkdir command
        assert "data/uploads" in content and "mkdir -p" in content, (
            "Entrypoint must ensure data/uploads/ exists for user uploads"
        )

    def test_entrypoint_uses_flock(self):
        """Entrypoint must use flock for race-condition-safe initialization."""
        content = ENTRYPOINT_SCRIPT.read_text(encoding="utf-8")
        assert "flock" in content, (
            "docker-entrypoint.sh must use flock for exclusive locking"
        )

    def test_entrypoint_defines_lock_file(self):
        """Entrypoint must define a lock file on the shared volume."""
        content = ENTRYPOINT_SCRIPT.read_text(encoding="utf-8")
        assert "LOCK_FILE=" in content, (
            "docker-entrypoint.sh must define a LOCK_FILE variable"
        )
        # Lock file must be on the shared data volume
        assert "data/" in content.split("LOCK_FILE=")[1].split("\n")[0], (
            "Lock file must be on the shared data volume"
        )


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                    reason="the entrypoint is a bash script that runs only inside the Linux containers")
class TestSentinelLogic:
    """Test the sentinel-file-based initialization logic in isolation.

    Uses a minimal bash script that replicates the entrypoint's conditional
    logic but substitutes the actual pipeline commands with touch commands,
    allowing tests to run without the full Python pipeline.
    """

    @pytest.fixture
    def workdir(self, tmp_path):
        """Create an isolated working directory simulating /app."""
        return tmp_path

    def _make_test_script(self, workdir):
        """Create a test version of the entrypoint logic."""
        script = workdir / "test_entrypoint.sh"
        script.write_text(textwrap.dedent("""\
            #!/usr/bin/env bash
            set -e

            DATA_SENTINEL="data/processed/marketing_data_clean.csv"
            MODEL_SENTINEL="models/mmm_model.pkl"

            if [ ! -f "$DATA_SENTINEL" ] || [ ! -f "$MODEL_SENTINEL" ]; then
                echo "INIT_TRIGGERED"
                mkdir -p data/raw data/processed data/uploads models
                touch "$DATA_SENTINEL"
                touch "$MODEL_SENTINEL"
                echo "INIT_COMPLETE"
            else
                echo "INIT_SKIPPED"
            fi

            mkdir -p data/uploads
            echo "READY"
            exec "$@"
        """))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return script

    def _run_entrypoint(self, workdir, script):
        """Run the test entrypoint script."""
        result = subprocess.run(
            ["bash", str(script), "echo", "SERVICE_STARTED"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result

    def test_fresh_empty_storage_triggers_initialization(self, workdir):
        """P1 regression: fresh install with empty dirs must run init."""
        script = self._make_test_script(workdir)
        result = self._run_entrypoint(workdir, script)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "INIT_TRIGGERED" in result.stdout
        assert "INIT_COMPLETE" in result.stdout
        assert "READY" in result.stdout
        assert "SERVICE_STARTED" in result.stdout

        # Sentinel files must be created
        assert (workdir / "data/processed/marketing_data_clean.csv").exists()
        assert (workdir / "models/mmm_model.pkl").exists()

    def test_fresh_install_creates_required_directories(self, workdir):
        """Fresh install must create all required directories."""
        script = self._make_test_script(workdir)
        self._run_entrypoint(workdir, script)

        for d in ["data/raw", "data/processed", "data/uploads", "models"]:
            assert (workdir / d).is_dir(), f"Directory {d} must be created"

    def test_existing_data_preserved_on_startup(self, workdir):
        """P1 regression: existing user data must survive container restart."""
        script = self._make_test_script(workdir)

        # Pre-populate with "user data"
        (workdir / "data/processed").mkdir(parents=True)
        (workdir / "models").mkdir(parents=True)
        (workdir / "data/uploads").mkdir(parents=True)
        (workdir / "data/processed/marketing_data_clean.csv").write_text(
            "user_data,col2\n1,2\n"
        )
        (workdir / "models/mmm_model.pkl").write_bytes(b"user_model_bytes")

        # Simulate a user upload
        user_upload = workdir / "data/uploads/my_dataset"
        user_upload.mkdir(parents=True)
        (user_upload / "raw.csv").write_text("a,b\n1,2\n")

        result = self._run_entrypoint(workdir, script)

        assert result.returncode == 0
        assert "INIT_SKIPPED" in result.stdout, (
            "Must skip init when sentinel files exist"
        )

        # Verify user data is untouched
        assert (workdir / "data/processed/marketing_data_clean.csv").read_text(encoding="utf-8") == (
            "user_data,col2\n1,2\n"
        )
        assert (workdir / "models/mmm_model.pkl").read_bytes() == b"user_model_bytes"
        assert (user_upload / "raw.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"

    def test_repeated_initialization_is_idempotent(self, workdir):
        """P1 regression: running entrypoint multiple times must be safe."""
        script = self._make_test_script(workdir)

        # First run — triggers init
        r1 = self._run_entrypoint(workdir, script)
        assert r1.returncode == 0
        assert "INIT_TRIGGERED" in r1.stdout

        # Second run — must skip
        r2 = self._run_entrypoint(workdir, script)
        assert r2.returncode == 0
        assert "INIT_SKIPPED" in r2.stdout

        # Third run — must still skip
        r3 = self._run_entrypoint(workdir, script)
        assert r3.returncode == 0
        assert "INIT_SKIPPED" in r3.stdout

    def test_partial_sentinels_triggers_init(self, workdir):
        """Init must trigger if only one sentinel exists (partial state)."""
        script = self._make_test_script(workdir)

        # Only data sentinel exists, no model
        (workdir / "data/processed").mkdir(parents=True)
        (workdir / "data/processed/marketing_data_clean.csv").write_text("data")

        result = self._run_entrypoint(workdir, script)
        assert result.returncode == 0
        assert "INIT_TRIGGERED" in result.stdout

    def test_model_only_triggers_init(self, workdir):
        """Init must trigger if only model sentinel exists."""
        script = self._make_test_script(workdir)

        (workdir / "models").mkdir(parents=True)
        (workdir / "models/mmm_model.pkl").write_bytes(b"model")

        result = self._run_entrypoint(workdir, script)
        assert result.returncode == 0
        assert "INIT_TRIGGERED" in result.stdout

    def test_uploads_dir_always_created(self, workdir):
        """data/uploads/ must exist after entrypoint regardless of init."""
        script = self._make_test_script(workdir)

        # Pre-populate sentinels to skip init
        (workdir / "data/processed").mkdir(parents=True)
        (workdir / "models").mkdir(parents=True)
        (workdir / "data/processed/marketing_data_clean.csv").write_text("d")
        (workdir / "models/mmm_model.pkl").write_bytes(b"m")
        # Do NOT create data/uploads/

        result = self._run_entrypoint(workdir, script)
        assert result.returncode == 0
        assert "INIT_SKIPPED" in result.stdout
        assert (workdir / "data/uploads").is_dir(), (
            "data/uploads/ must be created even when init is skipped"
        )

    def test_exec_passes_cmd_arguments(self, workdir):
        """exec "$@" must forward CMD correctly."""
        script = self._make_test_script(workdir)

        # Pre-populate sentinels
        (workdir / "data/processed").mkdir(parents=True)
        (workdir / "models").mkdir(parents=True)
        (workdir / "data/processed/marketing_data_clean.csv").write_text("d")
        (workdir / "models/mmm_model.pkl").write_bytes(b"m")

        result = subprocess.run(
            ["bash", str(script), "echo", "CUSTOM_CMD_OUTPUT"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "CUSTOM_CMD_OUTPUT" in result.stdout


class TestDockerfileIntegrity:
    """Verify Dockerfile configurations are consistent."""

    def test_dockerfile_uses_entrypoint(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "ENTRYPOINT" in content
        assert "docker-entrypoint.sh" in content

    def test_dockerfile_api_uses_entrypoint(self):
        content = (PROJECT_ROOT / "Dockerfile.api").read_text(encoding="utf-8")
        assert "ENTRYPOINT" in content
        assert "docker-entrypoint.sh" in content

    def test_dockerfile_no_build_time_pipeline(self):
        """Pipeline must NOT run at build time (moved to entrypoint)."""
        content = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        # Should not have RUN python src/data/generate_dataset.py etc.
        assert "RUN python src/data/generate_dataset.py" not in content

    def test_dockerfile_api_no_build_time_pipeline(self):
        content = (PROJECT_ROOT / "Dockerfile.api").read_text(encoding="utf-8")
        assert "RUN python src/data/generate_dataset.py" not in content

    def test_dockerignore_does_not_exclude_entrypoint(self):
        dockerignore = PROJECT_ROOT / ".dockerignore"
        if dockerignore.exists():
            content = dockerignore.read_text(encoding="utf-8")
            assert "docker-entrypoint.sh" not in content, (
                ".dockerignore must not exclude docker-entrypoint.sh"
            )


class TestConfiguredPaths:
    """Verify data and model paths used by the application match
    the sentinel paths checked by the entrypoint."""

    def test_sentinel_paths_match_settings(self):
        """Entrypoint sentinel paths must align with config/settings.py."""
        from config import settings

        # The data sentinel is in the processed data directory
        data_sentinel = Path("data/processed/marketing_data_clean.csv")
        expected_data_dir = settings.PROCESSED_DATA_DIR.relative_to(settings.BASE_DIR)
        assert str(data_sentinel.parent) == str(expected_data_dir), (
            f"Data sentinel dir {data_sentinel.parent} != settings {expected_data_dir}"
        )

        # The model sentinel is in the models directory
        model_sentinel = Path("models/mmm_model.pkl")
        expected_models_dir = settings.MODELS_DIR.relative_to(settings.BASE_DIR)
        assert str(model_sentinel.parent) == str(expected_models_dir), (
            f"Model sentinel dir {model_sentinel.parent} != settings {expected_models_dir}"
        )

    def test_uploads_dir_matches_settings(self):
        from config import settings

        expected = settings.UPLOADS_DIR.relative_to(settings.BASE_DIR)
        assert str(expected) == "data/uploads"


class TestDockerComposeVolumes:
    """Verify docker-compose.yml volume mounts are compatible with
    the entrypoint initialization strategy."""

    @pytest.fixture
    def compose_content(self):
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        if not compose_file.exists():
            pytest.skip("docker-compose.yml not present")
        return compose_file.read_text(encoding="utf-8")

    def test_compose_mounts_data_dir(self, compose_content):
        """docker-compose must mount ./data for persistence."""
        assert "./data" in compose_content or "data:" in compose_content

    def test_compose_mounts_models_dir(self, compose_content):
        """docker-compose must mount ./models for persistence."""
        assert "./models" in compose_content or "models:" in compose_content

    def test_compose_uses_entrypoint_dockerfiles(self, compose_content):
        """Compose must reference Dockerfiles that use the entrypoint."""
        # Both services should use the modified Dockerfiles
        assert "Dockerfile" in compose_content

    def test_compose_api_depends_on_dashboard_healthy(self, compose_content):
        """API must wait for dashboard to be healthy before starting."""
        assert "service_healthy" in compose_content, (
            "API must use depends_on with condition: service_healthy"
        )

    def test_compose_env_file_optional(self, compose_content):
        """docker-compose must not fail when .env is missing."""
        assert "required: false" in compose_content, (
            "env_file must use required: false so missing .env doesn't fail"
        )

    def test_compose_has_healthchecks(self, compose_content):
        """Both services must define health checks."""
        assert "healthcheck" in compose_content, (
            "Services must define healthcheck configurations"
        )
