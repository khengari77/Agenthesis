"""Invariant decorators for asserting agent behavioral properties."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from agentcheck._context import get_current_intercept, set_pending_limits
from agentcheck.types import InvariantViolation

if TYPE_CHECKING:
    from collections.abc import Callable


def max_steps(n: int) -> Callable[..., Any]:
    """Assert that the agent completes in at most n steps.

    Sets a runtime limit on the Intercept context so execution aborts
    immediately when exceeded, then also validates post-mortem as a safety net.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            set_pending_limits(max_steps=n)
            result = fn(*args, **kwargs)

            # Post-mortem safety net
            ctx = get_current_intercept()
            trace = ctx.trace
            if trace.steps > n:
                raise InvariantViolation(
                    invariant="max_steps",
                    message=f"Agent took {trace.steps} steps, max allowed is {n}",
                    trace=trace,
                )
            return result

        return wrapper

    return decorator


def never_calls(tool_name: str) -> Callable[..., Any]:
    """Assert that the named tool is never invoked."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            ctx = get_current_intercept()
            trace = ctx.trace
            forbidden_calls = [tc for tc in trace.tool_calls if tc.name == tool_name]
            if forbidden_calls:
                raise InvariantViolation(
                    invariant="never_calls",
                    message=(
                        f"Tool '{tool_name}' was called {len(forbidden_calls)} time(s), "
                        f"but should never be called"
                    ),
                    trace=trace,
                )
            return result

        return wrapper

    return decorator


def requires_before(tool_a: str, tool_b: str) -> Callable[..., Any]:
    """Assert that tool_a must be called before tool_b."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            ctx = get_current_intercept()
            trace = ctx.trace
            names = [tc.name for tc in trace.tool_calls]

            if tool_b in names:
                b_index = names.index(tool_b)
                if tool_a not in names[:b_index]:
                    raise InvariantViolation(
                        invariant="requires_before",
                        message=(
                            f"Tool '{tool_a}' must be called before '{tool_b}', "
                            f"but call order was: {names}"
                        ),
                        trace=trace,
                    )
            return result

        return wrapper

    return decorator


def max_llm_calls(n: int) -> Callable[..., Any]:
    """Assert that the agent makes at most n LLM calls.

    Sets a runtime limit so execution aborts immediately when exceeded.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            set_pending_limits(max_llm_calls=n)
            result = fn(*args, **kwargs)

            # Post-mortem safety net
            ctx = get_current_intercept()
            trace = ctx.trace
            if trace.llm_calls > n:
                raise InvariantViolation(
                    invariant="max_llm_calls",
                    message=f"Agent made {trace.llm_calls} LLM calls, max allowed is {n}",
                    trace=trace,
                )
            return result

        return wrapper

    return decorator


def max_token_cost(max_tokens: int) -> Callable[..., Any]:
    """Assert that total token usage stays within budget.

    Sets a runtime limit so execution aborts immediately when exceeded.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            set_pending_limits(max_tokens=max_tokens)
            result = fn(*args, **kwargs)

            # Post-mortem safety net
            ctx = get_current_intercept()
            trace = ctx.trace
            if trace.total_tokens > max_tokens:
                raise InvariantViolation(
                    invariant="max_token_cost",
                    message=(
                        f"Agent used {trace.total_tokens} tokens, "
                        f"max allowed is {max_tokens}"
                    ),
                    trace=trace,
                )
            return result

        return wrapper

    return decorator


def output_matches_schema(schema: dict[str, Any]) -> Callable[..., Any]:
    """Assert that the agent's output conforms to a JSON schema.

    Requires the 'json' optional dependency: pip install agentcheck[json]
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                import jsonschema
            except ImportError as e:
                msg = (
                    "jsonschema is required for output_matches_schema. "
                    "Install with: pip install agentcheck[json]"
                )
                raise ImportError(msg) from e

            result = fn(*args, **kwargs)

            # The result should be an AgentResult or have an 'output' attribute
            output = result.output if hasattr(result, "output") else result

            import json

            try:
                parsed = json.loads(output) if isinstance(output, str) else output
            except json.JSONDecodeError as e:
                ctx = get_current_intercept()
                raise InvariantViolation(
                    invariant="output_matches_schema",
                    message=f"Output is not valid JSON: {e}",
                    trace=ctx.trace,
                ) from e

            try:
                jsonschema.validate(parsed, schema)
            except jsonschema.ValidationError as e:
                ctx = get_current_intercept()
                raise InvariantViolation(
                    invariant="output_matches_schema",
                    message=f"Output does not match schema: {e.message}",
                    trace=ctx.trace,
                ) from e

            return result

        return wrapper

    return decorator
