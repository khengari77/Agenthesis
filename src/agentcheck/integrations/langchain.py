"""LangChain callback handler for AgentCheck instrumentation."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as e:
    msg = (
        "langchain-core is required for the LangChain integration. "
        "Install with: pip install agentcheck[langchain]"
    )
    raise ImportError(msg) from e

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

from agentcheck._context import get_current_intercept


class AgentCheckCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records events into the active Intercept context."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            ctx = get_current_intercept()
        except Exception:
            return

        total_tokens = 0
        if response.llm_output and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage", {})
            if isinstance(token_usage, dict):
                total_tokens = token_usage.get("total_tokens", 0)

        with contextlib.suppress(Exception):
            ctx.record_llm_call(tokens=total_tokens)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        with contextlib.suppress(Exception):
            ctx = get_current_intercept()
            ctx.record_step()
