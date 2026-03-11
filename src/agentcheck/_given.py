"""Thin wrapper around hypothesis.given with agent-friendly defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypothesis import HealthCheck, settings
from hypothesis import given as _hypothesis_given

if TYPE_CHECKING:
    from collections.abc import Callable

_AGENT_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def given(
    *args: Any,
    max_examples: int = 10,
    **kwargs: Any,
) -> Callable[..., Any]:
    """AgentCheck's @given decorator.

    Wraps hypothesis.given with agent-appropriate defaults:
    - max_examples=10 (agent calls are expensive)
    - deadline=None (agent calls are slow)
    - suppress too_slow health check
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        configured = settings(
            max_examples=max_examples,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow],
        )(fn)
        return _hypothesis_given(*args, **kwargs)(configured)

    return decorator
