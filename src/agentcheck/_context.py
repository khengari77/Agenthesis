"""Thread-local context stack for connecting Intercept to invariant decorators."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from agentcheck.types import AgentCheckError

if TYPE_CHECKING:
    from agentcheck.intercept import Intercept

_context_stack = threading.local()


def push_context(ctx: Intercept) -> None:
    """Push an Intercept context onto the thread-local stack."""
    if not hasattr(_context_stack, "stack"):
        _context_stack.stack = []
    _context_stack.stack.append(ctx)


def pop_context() -> Intercept:
    """Pop the most recent Intercept context from the stack.

    The popped context is saved as 'last' so invariant decorators
    can access it after the with-block exits.
    """
    stack: list[Intercept] = getattr(_context_stack, "stack", [])
    if not stack:
        msg = "No active Intercept context to pop"
        raise AgentCheckError(msg)
    ctx = stack.pop()
    _context_stack.last = ctx
    return ctx


def get_current_intercept() -> Intercept:
    """Get the currently active or most recently exited Intercept context.

    First checks the active stack, then falls back to the last exited context.
    This allows invariant decorators to work both inside and after a with-block.
    """
    stack: list[Intercept] = getattr(_context_stack, "stack", [])
    if stack:
        return stack[-1]

    last: Intercept | None = getattr(_context_stack, "last", None)
    if last is not None:
        return last

    msg = "No active Intercept context. Use 'with Intercept(agent) as ctx:' first."
    raise AgentCheckError(msg)
