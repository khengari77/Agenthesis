"""Invariant decorators for asserting agent behavioral properties."""

from __future__ import annotations

import functools
import json
import re
from typing import TYPE_CHECKING, Any

from agenthesis._context import (
    enter_decorator,
    exit_decorator,
    get_all_test_intercepts,
    set_pending_limits,
)
from agenthesis.types import InvariantViolation

if TYPE_CHECKING:
    from collections.abc import Callable

_MARKDOWN_JSON_RE = re.compile(
    r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)


def _strip_markdown_fences(text: str) -> str:
    match = _MARKDOWN_JSON_RE.match(text)
    return match.group(1).strip() if match else text.strip()


def max_steps(n: int) -> Callable[..., Any]:
    """Assert that the agent completes in at most n steps.

    Sets a runtime limit on the Intercept context so execution aborts
    immediately when exceeded, then also validates post-mortem as a safety net.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            enter_decorator()
            set_pending_limits(max_steps=n)
            try:
                result = fn(*args, **kwargs)

                # Post-mortem safety net — check ALL intercepts
                for ctx in get_all_test_intercepts():
                    trace = ctx.trace
                    if trace.steps > n:
                        raise InvariantViolation(
                            invariant="max_steps",
                            message=f"Agent took {trace.steps} steps, max allowed is {n}",
                            trace=trace,
                        )
                return result
            finally:
                exit_decorator()

        return wrapper

    return decorator


def never_calls(tool_name: str) -> Callable[..., Any]:
    """Assert that the named tool is never invoked."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            enter_decorator()
            try:
                result = fn(*args, **kwargs)

                for ctx in get_all_test_intercepts():
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
            finally:
                exit_decorator()

        return wrapper

    return decorator


def requires_before(tool_a: str, tool_b: str) -> Callable[..., Any]:
    """Assert that tool_a must be called before tool_b."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            enter_decorator()
            try:
                result = fn(*args, **kwargs)

                for ctx in get_all_test_intercepts():
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
            finally:
                exit_decorator()

        return wrapper

    return decorator


def max_llm_calls(n: int) -> Callable[..., Any]:
    """Assert that the agent makes at most n LLM calls.

    Sets a runtime limit so execution aborts immediately when exceeded.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            enter_decorator()
            set_pending_limits(max_llm_calls=n)
            try:
                result = fn(*args, **kwargs)

                # Post-mortem safety net — check ALL intercepts
                for ctx in get_all_test_intercepts():
                    trace = ctx.trace
                    if trace.llm_calls > n:
                        raise InvariantViolation(
                            invariant="max_llm_calls",
                            message=f"Agent made {trace.llm_calls} LLM calls, max allowed is {n}",
                            trace=trace,
                        )
                return result
            finally:
                exit_decorator()

        return wrapper

    return decorator


def max_token_cost(max_tokens: int) -> Callable[..., Any]:
    """Assert that total token usage stays within budget.

    Sets a runtime limit so execution aborts immediately when exceeded.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            enter_decorator()
            set_pending_limits(max_tokens=max_tokens)
            try:
                result = fn(*args, **kwargs)

                # Post-mortem safety net — check ALL intercepts
                for ctx in get_all_test_intercepts():
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
            finally:
                exit_decorator()

        return wrapper

    return decorator


def output_matches_schema(schema: dict[str, Any]) -> Callable[..., Any]:
    """Assert that the agent's output conforms to a JSON schema.

    Requires the 'json' optional dependency: pip install agenthesis[json]
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                import jsonschema
            except ImportError as e:
                msg = (
                    "jsonschema is required for output_matches_schema. "
                    "Install with: pip install agenthesis[json]"
                )
                raise ImportError(msg) from e

            enter_decorator()
            try:
                result = fn(*args, **kwargs)

                # The result should be an AgentResult or have an 'output' attribute
                output = result.output if hasattr(result, "output") else result

                try:
                    if isinstance(output, str):
                        parsed = json.loads(_strip_markdown_fences(output))
                    else:
                        parsed = output
                except json.JSONDecodeError as e:
                    intercepts = get_all_test_intercepts()
                    trace = intercepts[-1].trace if intercepts else None
                    raise InvariantViolation(
                        invariant="output_matches_schema",
                        message=f"Output is not valid JSON: {e}",
                        trace=trace,
                    ) from e

                try:
                    jsonschema.validate(parsed, schema)
                except jsonschema.ValidationError as e:
                    intercepts = get_all_test_intercepts()
                    trace = intercepts[-1].trace if intercepts else None
                    raise InvariantViolation(
                        invariant="output_matches_schema",
                        message=f"Output does not match schema: {e.message}",
                        trace=trace,
                    ) from e

                return result
            finally:
                exit_decorator()

        return wrapper

    return decorator


def output_matches_grammar(parser: Callable[[str], Any]) -> Callable[..., Any]:
    """Assert that the agent's output conforms to a grammar.

    Args:
        parser: Any callable that accepts a string and raises on invalid input.
            Can be a lark parser, a custom function, etc.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            enter_decorator()
            try:
                result = fn(*args, **kwargs)

                output = result.output if hasattr(result, "output") else result

                if not isinstance(output, str):
                    output = str(output)

                try:
                    parser(output)
                except Exception as e:
                    intercepts = get_all_test_intercepts()
                    trace = intercepts[-1].trace if intercepts else None
                    raise InvariantViolation(
                        invariant="output_matches_grammar",
                        message=f"Output does not match grammar: {e}",
                        trace=trace,
                    ) from e

                return result
            finally:
                exit_decorator()

        return wrapper

    return decorator


def lark_grammar(grammar_str: str, start: str = "start") -> Callable[[str], Any]:
    """Create a parser from a Lark grammar string.

    Requires the 'grammar' optional dependency: pip install agenthesis[grammar]
    """
    try:
        from lark import Lark
    except ImportError as e:
        msg = (
            "lark is required for lark_grammar. "
            "Install with: pip install agenthesis[grammar]"
        )
        raise ImportError(msg) from e

    lark_parser = Lark(grammar_str, start=start)
    return lark_parser.parse
