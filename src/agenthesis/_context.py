"""Context stack for connecting Intercept to invariant decorators.

Uses contextvars.ContextVar with immutable tuples for safe behavior
in both threaded and async contexts. Mutable structures (like lists)
would be shared across async tasks that inherit the same ContextVar
reference, corrupting the stack.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Generator
from typing import TYPE_CHECKING

from agenthesis.types import AgenthesisError

if TYPE_CHECKING:
    from agenthesis.intercept import Intercept

_context_stack: contextvars.ContextVar[tuple[Intercept, ...]] = contextvars.ContextVar(
    "_context_stack", default=()
)
_context_last: contextvars.ContextVar[Intercept | None] = contextvars.ContextVar(
    "_context_last", default=None
)
_pending_limits: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_pending_limits", default=None
)
_test_intercepts: contextvars.ContextVar[tuple[Intercept, ...]] = contextvars.ContextVar(
    "_test_intercepts", default=()
)
_decorator_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_decorator_depth", default=0
)


def push_context(ctx: Intercept) -> None:
    """Push an Intercept context onto the stack."""
    stack = _context_stack.get()
    _context_stack.set((*stack, ctx))


def pop_context() -> Intercept:
    """Pop the most recent Intercept context from the stack.

    The popped context is saved as 'last' so invariant decorators
    can access it after the with-block exits.
    """
    stack = _context_stack.get()
    if not stack:
        msg = "No active Intercept context to pop"
        raise AgenthesisError(msg)
    ctx = stack[-1]
    _context_stack.set(stack[:-1])
    _context_last.set(ctx)
    return ctx


def get_current_intercept() -> Intercept:
    """Get the currently active or most recently exited Intercept context.

    First checks the active stack, then falls back to the last exited context.
    This allows invariant decorators to work both inside and after a with-block.
    """
    stack = _context_stack.get()
    if stack:
        return stack[-1]

    last = _context_last.get()
    if last is not None:
        return last

    msg = "No active Intercept context. Use 'with Intercept(agent) as ctx:' first."
    raise AgenthesisError(msg)


def set_pending_limits(**limits: int) -> None:
    """Store limits to be ingested by Intercept.__enter__.

    Limits accumulate (multiple decorators can each add their own).
    """
    current = _pending_limits.get()
    if current is None:
        _pending_limits.set(dict(limits))
    else:
        _pending_limits.set({**current, **limits})


def read_pending_limits() -> dict[str, int] | None:
    """Read pending limits without clearing them.

    Called by Intercept.__enter__ so that multiple Intercept contexts
    within the same decorated test all inherit the same limits.
    """
    return _pending_limits.get()


def record_test_intercept(ctx: Intercept) -> None:
    """Append an exited Intercept to the test-scoped accumulator."""
    current = _test_intercepts.get()
    _test_intercepts.set((*current, ctx))


def get_all_test_intercepts() -> tuple[Intercept, ...]:
    """Return all Intercepts that exited during this test.

    Falls back to get_current_intercept() wrapped in a tuple
    if no intercepts have been recorded (backward compatibility).
    """
    recorded = _test_intercepts.get()
    if recorded:
        return recorded
    return (get_current_intercept(),)


def clear_test_state() -> None:
    """Clear all test-scoped state: pending limits and recorded intercepts."""
    _pending_limits.set(None)
    _test_intercepts.set(())


@contextlib.contextmanager
def decorator_scope() -> Generator[None, None, None]:
    """Context manager for invariant decorator scope.

    Uses ContextVar tokens to ensure state is properly restored even
    if the decorated test raises an exception. Only the outermost
    decorator scope clears test state on exit.
    """
    depth_token = _decorator_depth.set(_decorator_depth.get() + 1)
    try:
        yield
    finally:
        _decorator_depth.reset(depth_token)
        if _decorator_depth.get() == 0:
            clear_test_state()


def enter_decorator() -> None:
    """Increment decorator depth. Called at the start of each invariant decorator.

    .. deprecated:: Use :func:`decorator_scope` context manager instead.
    """
    _decorator_depth.set(_decorator_depth.get() + 1)


def exit_decorator() -> None:
    """Decrement decorator depth. Only the outermost decorator clears test state.

    .. deprecated:: Use :func:`decorator_scope` context manager instead.
    """
    depth = _decorator_depth.get() - 1
    _decorator_depth.set(depth)
    if depth == 0:
        clear_test_state()


# Backward-compatible alias
clear_pending_limits = clear_test_state
