"""Tests for LangChainResolver."""

from __future__ import annotations

from typing import Any

import pytest

try:
    from langchain_core.tools import BaseTool  # noqa: TCH002

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")


def _make_tool(name: str, fn: Any) -> BaseTool:
    """Create a minimal BaseTool for testing."""
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(func=fn, name=name, description=f"Test tool: {name}")


class _FakeAgent:
    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools


class TestLangChainResolver:
    def test_resolve_discovers_base_tools(self) -> None:
        from agentcheck.integrations.langchain.resolver import LangChainResolver

        t1 = _make_tool("search", lambda q="": f"found: {q}")
        t2 = _make_tool("calc", lambda x=0: x * 2)
        agent = _FakeAgent([t1, t2])

        resolver = LangChainResolver()
        tools = resolver.resolve(agent, None)
        assert "search" in tools
        assert "calc" in tools

    def test_resolve_explicit_tools(self) -> None:
        from agentcheck.integrations.langchain.resolver import LangChainResolver

        resolver = LangChainResolver()
        explicit = {"my_tool": lambda: 42}
        tools = resolver.resolve(None, explicit)
        assert tools["my_tool"]() == 42

    def test_install_patches_run(self) -> None:
        from agentcheck.integrations.langchain.resolver import LangChainResolver

        t = _make_tool("search", lambda q="": f"found: {q}")
        resolver = LangChainResolver(lc_tools=[t])
        resolver.resolve(None, None)  # populate _tool_map

        original_run = t._run

        def wrapper(q=""):  # noqa: ANN001
            return "wrapped"

        resolver.install(None, None, "search", wrapper)
        assert t._run is wrapper

        resolver.restore(None, None, "search", original_run)
        assert t._run is original_run

    def test_intercept_with_langchain_resolver(self) -> None:
        from agentcheck.integrations.langchain.resolver import LangChainResolver
        from agentcheck.intercept import Intercept

        t = _make_tool("search", lambda q="": f"found: {q}")
        resolver = LangChainResolver(lc_tools=[t])

        with Intercept(resolver=resolver) as ctx:
            ctx.on("search").respond("mocked")
            result = t._run(q="test")

        assert result == "mocked"
        assert len(ctx.calls) == 1
        assert ctx.calls[0].name == "search"
