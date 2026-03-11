"""Context stack for connecting Intercept to invariant decorators.

Uses contextvars.ContextVar for safe behavior in both threaded and async contexts.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

from agentcheck.types import AgentCheckError

if TYPE_CHECKING:
    from agentcheck.intercept import Intercept

_context_stack: contextvars.ContextVar[list[Intercept] | None] = contextvars.ContextVar(
    "_context_stack", default=None
)
_context_last: contextvars.ContextVar[Intercept | None] = contextvars.ContextVar(
    "_context_last", default=None
)
_pending_limits: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_pending_limits", default=None
)


def push_context(ctx: Intercept) -> None:
    """Push an Intercept context onto the stack."""
    stack = _context_stack.get()
    if stack is None:
        _context_stack.set([ctx])
    else:
        stack.append(ctx)


def pop_context() -> Intercept:
    """Pop the most recent Intercept context from the stack.

    The popped context is saved as 'last' so invariant decorators
    can access it after the with-block exits.
    """
    stack = _context_stack.get()
    if not stack:
        msg = "No active Intercept context to pop"
        raise AgentCheckError(msg)
    ctx = stack.pop()
    _context_last.set(ctx)
    return ctx


def get_current_intercept() -> Intercept:
    """Get the currently active or most recently exited Intercept context.

    First checks the active stack, then falls back to the last exited context.
    This allows invariant decorators to work both inside and after a with-block.
    """
    stack = _context_stack.get()
    if stack:  # handles both None and empty list
        return stack[-1]

    last = _context_last.get()
    if last is not None:
        return last

    msg = "No active Intercept context. Use 'with Intercept(agent) as ctx:' first."
    raise AgentCheckError(msg)


def set_pending_limits(**limits: int) -> None:
    """Store limits to be ingested by the next Intercept.__enter__."""
    current = _pending_limits.get()
    if current is None:
        _pending_limits.set(dict(limits))
    else:
        current.update(limits)


def consume_pending_limits() -> dict[str, int] | None:
    """Pop and return pending limits (called by Intercept.__enter__)."""
    limits = _pending_limits.get()
    _pending_limits.set(None)
    return limits
