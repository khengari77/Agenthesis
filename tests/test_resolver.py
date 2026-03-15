"""Tests for DefaultResolver and custom resolver integration."""

from __future__ import annotations

from typing import Any

from agentcheck.intercept import DefaultResolver, Intercept


class _ToolKitAgent:
    """Agent that implements the ToolKit protocol."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {
            "search": lambda q: f"found: {q}",
            "calc": lambda x: x * 2,
        }

    def get_tools(self) -> dict[str, Any]:
        return self._tools

    def set_tool(self, name: str, fn: Any) -> None:
        self._tools[name] = fn

    def run(self, prompt: str) -> Any:
        return None


class _AttrAgent:
    """Agent that exposes tool_* attributes."""

    def tool_search(self, q: str = "") -> str:
        return f"found: {q}"

    def tool_calc(self, x: int = 0) -> int:
        return x * 2


class TestDefaultResolver:
    def test_toolkit_agent(self) -> None:
        agent = _ToolKitAgent()
        resolver = DefaultResolver()
        tools = resolver.resolve(agent, None)
        assert "search" in tools
        assert "calc" in tools

    def test_attribute_agent(self) -> None:
        agent = _AttrAgent()
        resolver = DefaultResolver()
        tools = resolver.resolve(agent, None)
        assert "search" in tools
        assert "calc" in tools

    def test_explicit_tools(self) -> None:
        resolver = DefaultResolver()
        explicit = {"my_tool": lambda: 42}
        tools = resolver.resolve(None, explicit)
        assert "my_tool" in tools
        assert tools["my_tool"]() == 42

    def test_install_restore_roundtrip(self) -> None:
        agent = _ToolKitAgent()
        resolver = DefaultResolver()
        original_search = agent._tools["search"]

        def wrapper(q):  # noqa: ANN001
            return "wrapped"

        resolver.install(agent, None, "search", wrapper)
        assert agent._tools["search"] is wrapper

        resolver.restore(agent, None, "search", original_search)
        assert agent._tools["search"] is original_search


class _CustomResolver:
    """A custom ToolResolver that resolves from a fixed dict."""

    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools
        self._installed: dict[str, Any] = {}

    def resolve(self, agent: Any, explicit_tools: dict[str, Any] | None) -> dict[str, Any]:
        return dict(self._tools)

    def install(
        self, agent: Any, explicit_tools: dict[str, Any] | None, name: str, wrapper: Any,
    ) -> None:
        self._installed[name] = self._tools[name]
        self._tools[name] = wrapper

    def restore(
        self, agent: Any, explicit_tools: dict[str, Any] | None, name: str, original: Any,
    ) -> None:
        self._tools[name] = original


class TestCustomResolver:
    def test_custom_resolver_used_by_intercept(self) -> None:
        tools = {"echo": lambda x: x}
        resolver = _CustomResolver(tools)

        with Intercept(resolver=resolver) as ctx:
            ctx.on("echo").respond("mocked")
            result = tools["echo"]("hello")

        assert result == "mocked"
        # Verify install() saved the original
        assert "echo" in resolver._installed
        # After exit, original is restored
        assert tools["echo"]("hello") == "hello"

    def test_intercept_with_agent_tools_and_resolver(self) -> None:
        """Pass both agent and explicit tools with a custom resolver."""

        class _FakeAgent:
            def run(self, prompt: str) -> Any:
                return None

        agent = _FakeAgent()
        explicit = {"greet": lambda name: f"hi {name}"}
        resolver = _CustomResolver(explicit)

        with Intercept(agent, explicit, resolver=resolver) as ctx:
            ctx.on("greet").respond("mocked greeting")
            result = explicit["greet"]("world")

        assert result == "mocked greeting"
        assert len(ctx.calls) == 1
        assert ctx.calls[0].name == "greet"
        # Original restored
        assert explicit["greet"]("world") == "hi world"
