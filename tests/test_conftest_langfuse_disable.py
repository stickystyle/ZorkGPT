# ABOUTME: Test to verify conftest.py fixture disables Langfuse by default
# ABOUTME: Ensures test data doesn't pollute production Langfuse or consume quota

import pytest
import os


def test_langfuse_env_vars_disabled_by_default():
    """Verify that conftest.py fixture removes Langfuse env vars by default."""
    # These should be None because the global fixture deletes them
    assert os.getenv("LANGFUSE_PUBLIC_KEY") is None
    assert os.getenv("LANGFUSE_SECRET_KEY") is None
    assert os.getenv("LANGFUSE_HOST") is None


def test_langfuse_can_be_enabled_explicitly(monkeypatch):
    """Verify that tests can still opt-in to Langfuse if needed."""
    # Explicitly enable Langfuse for this test
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")

    # Verify env vars are set
    assert os.getenv("LANGFUSE_PUBLIC_KEY") == "test-key"
    assert os.getenv("LANGFUSE_SECRET_KEY") == "test-secret"
    assert os.getenv("LANGFUSE_HOST") == "https://test.langfuse.com"


def test_orchestrator_works_without_langfuse():
    """Verify orchestrator can initialize without Langfuse credentials."""
    from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
    from unittest.mock import patch

    # Since global fixture already cleared env vars, orchestrator should work fine
    with patch('orchestration.zork_orchestrator_v2.Langfuse', None):
        orchestrator = ZorkOrchestratorV2(episode_id="test-no-langfuse")

        # Should initialize successfully with langfuse_client = None
        assert orchestrator is not None
        assert orchestrator.langfuse_client is None
