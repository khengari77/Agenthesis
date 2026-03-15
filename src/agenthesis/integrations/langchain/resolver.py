"""LangChain-specific tool resolver."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from agenthesis.types import InterceptError


class LangChainResolver:
    """ToolResolver implementation for LangChain BaseTool instances."""

    def __init__(self, lc_tools: list[BaseTool] | None = None) -> None:
        self._lc_tools = lc_tools
        self._tool_map: dict[str, BaseTool] = {}

    def resolve(self, agent: Any, explicit_tools: dict[str, Any] | None) -> dict[str, Any]:
        if explicit_tools is not None:
            return dict(explicit_tools)

        tools: list[BaseTool] = []
        if self._lc_tools is not None:
            tools = self._lc_tools
        elif hasattr(agent, "tools"):
            tools = agent.tools

        result: dict[str, Any] = {}
        self._tool_map = {}
        for tool in tools:
            if isinstance(tool, BaseTool):
                self._tool_map[tool.name] = tool
                result[tool.name] = tool._run
        return result

    def install(
        self, agent: Any, explicit_tools: dict[str, Any] | None, name: str, wrapper: Any
    ) -> None:
        if explicit_tools is not None:
            explicit_tools[name] = wrapper
            return

        tool = self._tool_map.get(name)
        if tool is None:
            msg = (
                f"Cannot install wrapper for unknown tool: {name}. "
                "Ensure resolve() is called before install()."
            )
            raise InterceptError(msg)

        try:
            object.__setattr__(tool, "_run", wrapper)
        except (AttributeError, TypeError):
            tool._run = wrapper  # type: ignore[assignment]

    def restore(
        self, agent: Any, explicit_tools: dict[str, Any] | None, name: str, original: Any
    ) -> None:
        if explicit_tools is not None:
            explicit_tools[name] = original
            return

        tool = self._tool_map.get(name)
        if tool is None:
            return

        try:
            object.__setattr__(tool, "_run", original)
        except (AttributeError, TypeError):
            tool._run = original  # type: ignore[assignment]
