"""Thin wrapper around hypothesis.given with agent-friendly defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypothesis import HealthCheck, settings
from hypothesis.errors import InvalidArgument
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

    Users can override with @settings(...) applied *outside* @ac.given,
    which Hypothesis naturally treats as the winning configuration.
    Other Hypothesis tests in the same process are not affected.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        wrapped = _hypothesis_given(*args, **kwargs)(fn)
        # Apply agent defaults only if the user hasn't already applied
        # @settings. Hypothesis raises InvalidArgument on double-settings,
        # which we catch to gracefully defer to the user's configuration.
        try:
            return _AGENT_SETTINGS(wrapped)
        except InvalidArgument:
            return wrapped

    return decorator
