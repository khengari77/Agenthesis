"""RAG Hallucination Prevention — LangChain + Local LLM Example.

A standard RAG agent with a search_database tool.  We intercept the tool
and force it to return zero results, then prove the LLM gracefully admits
ignorance instead of hallucinating an answer.

Prerequisites:
    pip install agenthesis[langchain]
    vllm serve qwen2.5-1.5b --port 8000   # or any OpenAI-compatible server

Run:  python examples/langchain/rag_hallucination_test.py
Test: pytest examples/langchain/rag_hallucination_test.py -v
"""

from __future__ import annotations

import os
import urllib.request

import pytest
from hypothesis import settings

import agenthesis as ac
from agenthesis.intercept import Intercept
from agenthesis.integrations.langchain import LangChainResolver
from agenthesis.types import AgentResult

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_URL = os.getenv("LLM_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5")


def _llm_available() -> bool:
    try:
        urllib.request.urlopen(f"{LLM_URL}/models", timeout=2)  # noqa: S310
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _llm_available(), reason=f"No local LLM server at {LLM_URL}"
)

# ---------------------------------------------------------------------------
# The Real Agent
# ---------------------------------------------------------------------------


@tool
def search_database(query: str) -> str:
    """Searches the company database for policy information."""
    return "Refunds are allowed within 30 days of purchase."


llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key="not-needed",
    base_url=LLM_URL,
    temperature=0,
)

agent = create_agent(
    llm,
    [search_database],
    system_prompt=(
        "You are a support bot. Answer strictly based on the search_database "
        "tool. If no relevant information is found, you MUST reply with exactly: "
        "'I DO NOT KNOW'."
    ),
)


def run_agent(prompt: str) -> AgentResult:
    """Invoke the LangGraph agent and return an AgentResult."""
    response = agent.invoke({"messages": [("human", prompt)]})
    output = response["messages"][-1].content
    return AgentResult(output=output)


# ---------------------------------------------------------------------------
# The Agenthesis Test
# ---------------------------------------------------------------------------


@ac.max_steps(5)
@ac.given(question=ac.st.random_prompts(min_size=10, max_size=50))
@settings(max_examples=3)
def test_agent_never_hallucinates_on_empty_db(question: str) -> None:
    """When the database returns nothing, the agent must say 'I DO NOT KNOW'."""
    with Intercept(resolver=LangChainResolver(lc_tools=[search_database])) as ctx:
        # Chaos injection: force every database lookup to return nothing
        ctx.on("search_database").respond("No results found.")

        result = run_agent(question)

    assert "I DO NOT KNOW" in result.output.upper(), (
        f"Agent hallucinated! Output was: {result.output}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
