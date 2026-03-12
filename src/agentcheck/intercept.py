"""Intercept context manager for hijacking agent tool calls."""

from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING, Any

from agentcheck._context import (
    pop_context,
    push_context,
    read_pending_limits,
    record_test_intercept,
)
from agentcheck.types import AgentTrace, InterceptError, InvariantViolation, ToolCall, ToolKit

if TYPE_CHECKING:
    from collections.abc import Callable


class ToolStub:
    """Configures how an intercepted tool behaves."""

    def __init__(self, original: Callable[..., Any] | None = None) -> None:
        self._original = original
        self._mode: str = "passthrough"
        self._value: Any = None
        self._sequence: list[Any] = []
        self._sequence_index: int = 0
        self._fn: Callable[..., Any] | None = None
        self._exception: Exception | None = None

    def respond(self, value: Any) -> ToolStub:
        """Return a fixed value when the tool is called."""
        self._mode = "respond"
        self._value = value
        return self

    def respond_with(self, fn: Callable[..., Any]) -> ToolStub:
        """Return the result of calling fn with the tool's arguments."""
        self._mode = "respond_with"
        self._fn = fn
        return self

    def raise_error(self, exc: Exception) -> ToolStub:
        """Raise an exception when the tool is called."""
        self._mode = "raise_error"
        self._exception = exc
        return self

    def passthrough(self) -> ToolStub:
        """Call the original tool (default behavior)."""
        self._mode = "passthrough"
        return self

    def respond_sequence(self, values: list[Any]) -> ToolStub:
        """Rotate through a sequence of values on successive calls."""
        self._mode = "respond_sequence"
        self._sequence = values
        self._sequence_index = 0
        return self

    def _execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the stub according to its configured mode."""
        if self._mode == "respond":
            return self._value
        if self._mode == "respond_with":
            if self._fn is None:
                msg = "Stub configured for 'respond_with' but no function was provided"
                raise InterceptError(msg)
            return self._fn(*args, **kwargs)
        if self._mode == "raise_error":
            if self._exception is None:
                msg = "Stub configured for 'raise_error' but no exception was provided"
                raise InterceptError(msg)
            raise self._exception
        if self._mode == "respond_sequence":
            if not self._sequence:
                msg = "respond_sequence called with empty sequence"
                raise InterceptError(msg)
            value = self._sequence[self._sequence_index % len(self._sequence)]
            self._sequence_index += 1
            return value
        # passthrough
        if self._original is not None:
            return self._original(*args, **kwargs)
        msg = "No original tool to passthrough to"
        raise InterceptError(msg)


class Intercept:
    """Context manager for intercepting agent tool calls.

    Usage:
        with Intercept(agent) as ctx:
            ctx.on("search_web").respond({"results": []})
            result = agent.run("find something")

        assert ctx.trace.steps <= 5
    """

    def __init__(
        self,
        agent: Any | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self._agent = agent
        self._explicit_tools = tools
        self._stubs: dict[str, ToolStub] = {}
        self._calls: list[ToolCall] = []
        self._originals: dict[str, Callable[..., Any]] = {}
        self._trace: AgentTrace | None = None
        self._start_time: float = 0.0
        self._step_counter: int = 0
        self._llm_call_counter: int = 0
        self._token_counter: int = 0
        self._max_steps: int | None = None
        self._max_tokens: int | None = None
        self._max_llm_calls: int | None = None

    def on(self, tool_name: str) -> ToolStub:
        """Configure stub behavior for a named tool."""
        original = self._originals.get(tool_name)
        stub = ToolStub(original)
        self._stubs[tool_name] = stub
        return stub

    def set_step_limit(self, n: int) -> None:
        """Set a runtime step limit that aborts execution immediately."""
        self._max_steps = n

    def set_token_limit(self, n: int) -> None:
        """Set a runtime token limit that aborts execution immediately."""
        self._max_tokens = n

    def set_llm_call_limit(self, n: int) -> None:
        """Set a runtime LLM call limit that aborts execution immediately."""
        self._max_llm_calls = n

    def record_step(self) -> None:
        """Manually record an agent step (for agents that report steps)."""
        self._step_counter += 1
        if self._max_steps is not None and self._step_counter > self._max_steps:
            raise InvariantViolation(
                invariant="max_steps",
                message=f"Agent took {self._step_counter} steps, max allowed is {self._max_steps}",
                trace=self._build_trace(),
            )

    def record_llm_call(self, tokens: int = 0) -> None:
        """Manually record an LLM call with optional token count."""
        self._llm_call_counter += 1
        self._token_counter += tokens
        if self._max_llm_calls is not None and self._llm_call_counter > self._max_llm_calls:
            raise InvariantViolation(
                invariant="max_llm_calls",
                message=(
                    f"Agent made {self._llm_call_counter} LLM calls, "
                    f"max allowed is {self._max_llm_calls}"
                ),
                trace=self._build_trace(),
            )
        if self._max_tokens is not None and self._token_counter > self._max_tokens:
            raise InvariantViolation(
                invariant="max_token_cost",
                message=(
                    f"Agent used {self._token_counter} tokens, "
                    f"max allowed is {self._max_tokens}"
                ),
                trace=self._build_trace(),
            )

    @property
    def trace(self) -> AgentTrace:
        """Get the execution trace. Available after context exit."""
        if self._trace is not None:
            return self._trace
        return self._build_trace()

    @property
    def calls(self) -> list[ToolCall]:
        """Get recorded tool calls (available during execution)."""
        return list(self._calls)

    def _build_trace(self) -> AgentTrace:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        return AgentTrace(
            tool_calls=tuple(self._calls),
            llm_calls=self._llm_call_counter,
            total_tokens=self._token_counter,
            steps=self._step_counter if self._step_counter else len(self._calls),
            duration_seconds=elapsed,
        )

    def _make_wrapper(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Create a recording wrapper for a tool (sync or async)."""
        intercept = self

        def _record_call(
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            result: Any,
            was_intercepted: bool,
        ) -> Any:
            """Shared recording and limit-checking logic."""
            call = ToolCall(
                name=name,
                arguments={"args": args, "kwargs": kwargs},
                result=result,
                timestamp=time.monotonic(),
                was_intercepted=was_intercepted,
            )
            intercept._calls.append(call)

            # Runtime step enforcement: when no manual record_step() is used,
            # steps defaults to len(calls), so check the limit here too.
            if (
                intercept._max_steps is not None
                and intercept._step_counter == 0
                and len(intercept._calls) > intercept._max_steps
            ):
                raise InvariantViolation(
                    invariant="max_steps",
                    message=(
                        f"Agent took {len(intercept._calls)} steps, "
                        f"max allowed is {intercept._max_steps}"
                    ),
                    trace=intercept._build_trace(),
                )

            return result

        if inspect.iscoroutinefunction(original):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                stub = intercept._stubs.get(name)
                was_intercepted = stub is not None and stub._mode != "passthrough"

                if stub is not None:
                    result = stub._execute(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                else:
                    result = await original(*args, **kwargs)

                _record_call(args, kwargs, result, was_intercepted)
                return result

            return async_wrapper

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            stub = intercept._stubs.get(name)
            was_intercepted = stub is not None and stub._mode != "passthrough"

            if stub is not None:
                result = stub._execute(*args, **kwargs)
            else:
                result = original(*args, **kwargs)

            _record_call(args, kwargs, result, was_intercepted)
            return result

        return wrapper

    def _resolve_tools(self) -> dict[str, Callable[..., Any]]:
        """Resolve the tools to intercept from agent or explicit dict."""
        if self._explicit_tools is not None:
            return dict(self._explicit_tools)

        if self._agent is not None and isinstance(self._agent, ToolKit):
            return self._agent.get_tools()

        if self._agent is not None:
            # Try attribute-based discovery
            tools: dict[str, Callable[..., Any]] = {}
            for attr_name in dir(self._agent):
                if attr_name.startswith("tool_"):
                    attr = getattr(self._agent, attr_name)
                    if callable(attr):
                        tool_name = attr_name[5:]  # strip "tool_" prefix
                        tools[tool_name] = attr
            if tools:
                return tools

        return {}

    def __enter__(self) -> Intercept:
        self._start_time = time.monotonic()
        self._calls.clear()
        self._step_counter = 0
        self._llm_call_counter = 0
        self._token_counter = 0
        self._trace = None

        # Resolve and wrap tools
        tools = self._resolve_tools()
        for name, fn in tools.items():
            self._originals[name] = fn
            wrapped = self._make_wrapper(name, fn)

            # Install the wrapper
            if self._explicit_tools is not None:
                self._explicit_tools[name] = wrapped
            elif self._agent is not None and isinstance(self._agent, ToolKit):
                self._agent.set_tool(name, wrapped)
            elif self._agent is not None and hasattr(self._agent, f"tool_{name}"):
                try:
                    setattr(self._agent, f"tool_{name}", wrapped)
                except AttributeError as e:
                    msg = (
                        f"Cannot intercept tool_{name} on {type(self._agent).__name__}: "
                        f"class uses __slots__ without a 'tool_{name}' slot"
                    )
                    raise InterceptError(msg) from e

        push_context(self)

        # Ingest any limits set by invariant decorators before this context opened.
        # Read (not consume) so that multiple Intercepts in the same test all get the limits.
        limits = read_pending_limits()
        if limits:
            if "max_steps" in limits:
                self._max_steps = limits["max_steps"]
            if "max_tokens" in limits:
                self._max_tokens = limits["max_tokens"]
            if "max_llm_calls" in limits:
                self._max_llm_calls = limits["max_llm_calls"]

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore original tools
        for name, original in self._originals.items():
            if self._explicit_tools is not None:
                self._explicit_tools[name] = original
            elif self._agent is not None and isinstance(self._agent, ToolKit):
                self._agent.set_tool(name, original)
            elif self._agent is not None and hasattr(self._agent, f"tool_{name}"):
                attr_name = f"tool_{name}"
                if attr_name in getattr(self._agent, "__dict__", {}):
                    # Instance __dict__ exists: remove wrapper so class descriptor is restored
                    delattr(self._agent, attr_name)
                else:
                    # __slots__ or no __dict__: restore original value directly
                    setattr(self._agent, attr_name, original)

        self._trace = self._build_trace()
        record_test_intercept(self)
        pop_context()
