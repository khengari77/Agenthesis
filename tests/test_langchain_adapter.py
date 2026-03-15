"""Tests for LangChainAgentAdapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    import langchain_core  # noqa: F401

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")


class TestLangChainAgentAdapter:
    def test_run_returns_agent_result(self) -> None:
        from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter
        from agenthesis.types import AgentResult

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "hello world"}
        mock_agent.tools = []

        adapter = LangChainAgentAdapter(mock_agent)
        result = adapter.run("test prompt")

        assert isinstance(result, AgentResult)
        assert result.output == "hello world"
        mock_agent.invoke.assert_called_once()

    def test_run_builds_trace_from_callback(self) -> None:
        from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "done"}
        mock_agent.tools = []

        adapter = LangChainAgentAdapter(mock_agent)
        result = adapter.run("test")

        # With no real LLM calls, counters should be 0
        assert result.trace.llm_calls == 0
        assert result.trace.total_tokens == 0

    def test_tools_property_proxies(self) -> None:
        from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter

        sentinel = [MagicMock(), MagicMock()]
        mock_agent = MagicMock()
        mock_agent.tools = sentinel

        adapter = LangChainAgentAdapter(mock_agent)
        assert adapter.tools is sentinel

    def test_extract_output_dict(self) -> None:
        from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter

        assert LangChainAgentAdapter._extract_output({"output": "hi"}) == "hi"

    def test_extract_output_string(self) -> None:
        from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter

        assert LangChainAgentAdapter._extract_output("plain") == "plain"

    def test_extract_output_message(self) -> None:
        from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter

        msg = MagicMock()
        msg.content = "from message"
        assert LangChainAgentAdapter._extract_output(msg) == "from message"

    def test_extract_output_missing_key_raises(self) -> None:
        from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter

        with pytest.raises(KeyError, match="Expected 'output' key"):
            LangChainAgentAdapter._extract_output({"result": "oops"})

    def test_custom_input_key(self) -> None:
        from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "done"}
        mock_agent.tools = []

        adapter = LangChainAgentAdapter(mock_agent, input_key="question")
        adapter.run("what is 2+2?")

        call_args = mock_agent.invoke.call_args
        assert "question" in call_args[0][0]
        assert call_args[0][0]["question"] == "what is 2+2?"
