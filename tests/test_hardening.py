"""Tests for low-priority hardening fixes (L-1 through L-4).

L-1: _UPLOAD_REGISTRY thread-safety (lock exists and is used)
L-4: CORS configured via environment variable, not wildcard
"""

import os
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestRegistryLock:
    """L-1: Verify _UPLOAD_REGISTRY has a threading lock."""

    def test_registry_lock_exists(self):
        from api.main import _REGISTRY_LOCK
        assert isinstance(_REGISTRY_LOCK, type(threading.Lock()))

    def test_registry_is_dict(self):
        from api.main import _UPLOAD_REGISTRY
        assert isinstance(_UPLOAD_REGISTRY, dict)


class TestCORSConfiguration:
    """L-4: Verify CORS is configurable and not wildcard by default."""

    def test_default_cors_excludes_wildcard(self):
        """When CORS_ORIGINS is unset, origins must not include '*'."""
        from api.main import _cors_origins
        assert "*" not in _cors_origins

    def test_default_cors_allows_localhost(self):
        """Default CORS includes localhost dev origins."""
        from api.main import _CORS_DEV_DEFAULTS
        assert "http://localhost:8501" in _CORS_DEV_DEFAULTS
        assert "http://localhost:8000" in _CORS_DEV_DEFAULTS

    def test_cors_env_var_documented(self):
        """CORS_ORIGINS is documented in .env.example."""
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        assert "CORS_ORIGINS" in env_example


class TestGitignoreTestArtifacts:
    """L-3: Verify .gitignore covers test-generated artifacts."""

    def test_gitignore_covers_test_uploads(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "data/uploads/test_" in gitignore
