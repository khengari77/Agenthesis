"""Thin wrapper around hypothesis.given with agent-friendly defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypothesis import HealthCheck, settings
from hypothesis import given as _hypothesis_given

if TYPE_CHECKING:
    from collections.abc import Callable

# Agent-friendly Hypothesis settings applied per-function (not globally).
_AGENT_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def given(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    """Agenthesis's @given decorator.

    Wraps hypothesis.given with agent-appropriate defaults applied
    only to the decorated function:
    - max_examples=10 (agent calls are expensive)
    - deadline=None (agent calls are slow)
    - suppress too_slow health check

    Users can override with @settings(...) on their test function.
    Other Hypothesis tests in the same process are not affected.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Check if the function already has explicit @settings applied.
        has_explicit_settings = hasattr(fn, "_hypothesis_internal_use_settings")
        wrapped = _hypothesis_given(*args, **kwargs)(fn)
        # Only apply agent defaults if the user hasn't set custom settings.
        if has_explicit_settings:
            return wrapped
        return _AGENT_SETTINGS(wrapped)

    return decorator
