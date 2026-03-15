"""LangChain agent adapter for AgentCheck."""

from __future__ import annotations

from typing import Any

from agentcheck.integrations.langchain.callback import AgentCheckCallbackHandler
from agentcheck.types import AgentResult, AgentTrace


class LangChainAgentAdapter:
    """Translates LangChain .invoke() into AgentResult."""

    def __init__(self, agent: Any, *, input_key: str = "input") -> None:
        self._agent = agent
        self._input_key = input_key
        self._handler = AgentCheckCallbackHandler()

    @property
    def tools(self) -> list[Any]:
        """Proxy agent.tools for resolver discovery."""
        return getattr(self._agent, "tools", [])

    def run(self, prompt: str) -> AgentResult:
        """Invoke the LangChain agent and return an AgentResult."""
        self._handler.reset()

        response = self._agent.invoke(
            {self._input_key: prompt},
            config={"callbacks": [self._handler]},
        )

        output = self._extract_output(response)
        counters = self._handler.get_trace()

        trace = AgentTrace(
            llm_calls=counters["llm_calls"],
            total_tokens=counters["total_tokens"],
            steps=counters["tool_calls"],
        )

        return AgentResult(output=output, trace=trace)

    @staticmethod
    def _extract_output(response: Any) -> str:
        if isinstance(response, dict):
            if "output" in response:
                return str(response["output"])
            msg = f"Expected 'output' key in response dict, got keys: {list(response.keys())}"
            raise KeyError(msg)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)
