"""Thin wrapper around hypothesis.given with agent-friendly defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypothesis import HealthCheck, settings
from hypothesis import given as _hypothesis_given

if TYPE_CHECKING:
    from collections.abc import Callable


def given(
    *args: Any,
    max_examples: int = 10,
    deadline: int | None = None,
    suppress_health_check: list[HealthCheck] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """AgentCheck's @given decorator.

    Wraps hypothesis.given with agent-appropriate defaults:
    - max_examples=10 (agent calls are expensive)
    - deadline=None (agent calls are slow)
    - suppress too_slow health check

    Users can override any setting, or apply their own @settings decorator
    below @ac.given (inner @settings takes precedence in Hypothesis).
    """
    if suppress_health_check is not None:
        health_checks = suppress_health_check
    else:
        health_checks = [HealthCheck.too_slow]

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        configured = settings(
            max_examples=max_examples,
            deadline=deadline,
            suppress_health_check=health_checks,
        )(fn)
        return _hypothesis_given(*args, **kwargs)(configured)

    return decorator
