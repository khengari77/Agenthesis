"""Thin wrapper around hypothesis.given with agent-friendly defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypothesis import HealthCheck, settings
from hypothesis import given as _hypothesis_given

if TYPE_CHECKING:
    from collections.abc import Callable

# Register and load an agent-friendly Hypothesis profile.
# Tests without an explicit @settings get these defaults automatically.
# Users can override per-test with @settings(...) or globally by loading
# a different profile after importing agentcheck.
settings.register_profile(
    "agentcheck",
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("agentcheck")


def given(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    """AgentCheck's @given decorator.

    Wraps hypothesis.given with agent-appropriate defaults loaded via
    the 'agentcheck' Hypothesis profile:
    - max_examples=10 (agent calls are expensive)
    - deadline=None (agent calls are slow)
    - suppress too_slow health check

    Users can override with @settings(...) on their test function,
    or by loading a different Hypothesis profile after importing agentcheck.
    """
    return _hypothesis_given(*args, **kwargs)
