"""Dummy agent for dogfooding AgentCheck."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agentcheck.types import AgentResult, AgentTrace, ToolCall

if TYPE_CHECKING:
    from collections.abc import Callable


class DummyAgent:
    """A deterministic test agent for dogfooding AgentCheck.

    Has three built-in tools: calculator, search, weather.
    Parses the prompt for keywords to decide which tools to call.

    Args:
        tools: Optional custom tools dict. If not provided, uses built-in tools.
        max_steps: Maximum number of steps before stopping.
        failure_mode: Optional mode to simulate misbehavior:
            - "infinite_loop": Ignores max_steps and loops
            - "wrong_tool": Calls forbidden tools
            - "token_explosion": Simulates excessive token usage
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        max_steps: int = 10,
        failure_mode: str | None = None,
    ) -> None:
        self._max_steps = max_steps
        self._failure_mode = failure_mode
        self._tool_calls: list[ToolCall] = []

        if tools is not None:
            self._tools = dict(tools)
        else:
            self._tools = {
                "calculator": self._builtin_calculator,
                "search": self._builtin_search,
                "weather": self._builtin_weather,
            }

    def get_tools(self) -> dict[str, Callable[..., Any]]:
        """Return the tools dict (implements ToolKit protocol)."""
        return self._tools

    def set_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Set a tool by name (implements ToolKit protocol)."""
        self._tools[name] = fn

    def run(self, prompt: str) -> AgentResult:
        """Run the agent with the given prompt."""
        start = time.monotonic()
        self._tool_calls = []
        steps = 0
        output_parts: list[str] = []

        limit = self._max_steps + 100 if self._failure_mode == "infinite_loop" else self._max_steps

        prompt_lower = prompt.lower() if prompt else ""

        for _ in range(limit):
            steps += 1

            # Decide which tool to call based on prompt keywords
            if self._failure_mode == "wrong_tool":
                # Intentionally call a forbidden tool
                tool_name = "execute_refund"
                if tool_name in self._tools:
                    result = self._tools[tool_name]()
                    self._record_call(tool_name, {}, result)
                    output_parts.append(f"Called {tool_name}: {result}")
                break

            if self._failure_mode == "token_explosion":
                # Simulate excessive processing
                trace = AgentTrace(
                    tool_calls=tuple(self._tool_calls),
                    llm_calls=100,
                    total_tokens=1_000_000,
                    steps=steps,
                    duration_seconds=time.monotonic() - start,
                )
                return AgentResult(
                    output="Token explosion simulated",
                    trace=trace,
                )

            if "calculate" in prompt_lower or "math" in prompt_lower:
                result = self._tools["calculator"](expression="2+2")
                self._record_call("calculator", {"expression": "2+2"}, result)
                output_parts.append(f"Calculator: {result}")
                break
            elif "search" in prompt_lower or "find" in prompt_lower:
                result = self._tools["search"](query=prompt)
                self._record_call("search", {"query": prompt}, result)
                output_parts.append(f"Search: {result}")
                break
            elif "weather" in prompt_lower:
                result = self._tools["weather"](location="default")
                self._record_call("weather", {"location": "default"}, result)
                output_parts.append(f"Weather: {result}")
                break
            else:
                # No keyword match, just respond
                output_parts.append(f"I received: {prompt[:100] if prompt else '(empty)'}")
                break

        elapsed = time.monotonic() - start
        trace = AgentTrace(
            tool_calls=tuple(self._tool_calls),
            llm_calls=1,
            total_tokens=len(prompt or "") * 2,
            steps=steps,
            duration_seconds=elapsed,
        )

        return AgentResult(
            output=" | ".join(output_parts) if output_parts else "",
            trace=trace,
        )

    def _record_call(self, name: str, args: dict[str, Any], result: Any) -> None:
        self._tool_calls.append(
            ToolCall(
                name=name,
                arguments=args,
                result=result,
                timestamp=time.monotonic(),
                was_intercepted=False,
            )
        )

    @staticmethod
    def _builtin_calculator(expression: str = "0") -> dict[str, Any]:
        allowed = set("0123456789+-*/ ()")
        if not all(c in allowed for c in expression):
            return {"result": "error", "expression": expression}
        try:
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        except Exception:
            result = "error"
        return {"result": result, "expression": expression}

    @staticmethod
    def _builtin_search(query: str = "") -> dict[str, Any]:
        return {"results": [{"title": "Result 1", "url": "https://example.com"}], "query": query}

    @staticmethod
    def _builtin_weather(location: str = "unknown") -> dict[str, Any]:
        return {"temperature": 22, "condition": "sunny", "location": location}
