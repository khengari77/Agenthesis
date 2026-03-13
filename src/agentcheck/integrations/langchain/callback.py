"""LangChain callback handler for AgentCheck instrumentation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from agentcheck._context import get_current_intercept
from agentcheck.types import AgentCheckError, ToolCall

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.outputs import LLMResult


class AgentCheckCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records events into the active Intercept context."""

    def __init__(self) -> None:
        super().__init__()
        self._llm_calls: int = 0
        self._total_tokens: int = 0
        self._tool_calls: int = 0
        self._pending_tools: dict[UUID, dict[str, Any]] = {}

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            ctx = get_current_intercept()
        except AgentCheckError:
            return  # No active context

        total_tokens = 0
        if response.llm_output and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage", {})
            if isinstance(token_usage, dict):
                total_tokens = token_usage.get("total_tokens", 0)

        self._llm_calls += 1
        self._total_tokens += total_tokens
        # InvariantViolation propagates — NOT inside try/except
        ctx.record_llm_call(tokens=total_tokens)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            ctx = get_current_intercept()
        except AgentCheckError:
            return  # No active context

        if run_id is not None:
            self._pending_tools[run_id] = {
                "name": serialized.get("name", "unknown"),
                "input": input_str,
                "start_time": time.monotonic(),
            }

        self._tool_calls += 1
        # InvariantViolation propagates — NOT inside try/except
        ctx.record_step()

    def on_tool_end(self, output: str, *, run_id: UUID | None = None, **kwargs: Any) -> None:
        try:
            ctx = get_current_intercept()
        except AgentCheckError:
            return

        if run_id is not None and run_id in self._pending_tools:
            info = self._pending_tools.pop(run_id)
            tool_name = info["name"]
            tool_input = info["input"]
        else:
            tool_name = "unknown"
            tool_input = ""

        call = ToolCall(
            name=tool_name,
            arguments={"input": tool_input},
            result=output,
            timestamp=time.monotonic(),
            was_intercepted=False,
        )
        ctx._calls.append(call)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._pending_tools.pop(run_id, None)

    def get_trace(self) -> dict[str, int]:
        """Return counter dict for adapter use."""
        return {
            "llm_calls": self._llm_calls,
            "total_tokens": self._total_tokens,
            "tool_calls": self._tool_calls,
        }

    def reset(self) -> None:
        """Clear all state for reuse."""
        self._llm_calls = 0
        self._total_tokens = 0
        self._tool_calls = 0
        self._pending_tools.clear()
