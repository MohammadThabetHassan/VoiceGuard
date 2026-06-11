"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear slowapi rate-limit counters between tests so the shared per-test
    /token logins (auth_client fixture) and burst tests don't bleed across tests."""
    from voiceguard.api.middleware import limiter

    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(autouse=True)
def _clear_result_store():
    """Isolate the in-memory detection-result store between tests."""
    from voiceguard.forensics import result_store

    result_store.clear()
    yield
    result_store.clear()
