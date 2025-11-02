# ABOUTME: Global pytest configuration for all tests
# ABOUTME: Disables Langfuse tracing by default to prevent test data pollution

import pytest


@pytest.fixture(autouse=True)
def disable_langfuse_for_tests(monkeypatch):
    """
    Disable Langfuse tracing for all tests by default.

    This prevents test data from polluting production Langfuse
    and consuming quota. Individual tests can opt-in to Langfuse
    by explicitly setting environment variables using monkeypatch.

    Example opt-in:
        def test_with_langfuse(monkeypatch):
            monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
            monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
            monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            # Test code here
    """
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
