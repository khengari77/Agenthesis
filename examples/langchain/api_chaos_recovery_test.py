"""API Chaos Recovery — LangChain + Local LLM Example.

An agent fetches a user profile from an external API.  We randomly inject
HTTP 500 and 429 errors into the tool and prove the LLM communicates the
failure to the user without crashing or looping infinitely.

Prerequisites:
    pip install agenthesis[langchain]
    vllm serve qwen2.5-1.5b --port 8000   # or any OpenAI-compatible server

Run:  python examples/langchain/api_chaos_recovery_test.py
Test: pytest examples/langchain/api_chaos_recovery_test.py -v
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
def fetch_user_profile(user_id: str) -> dict:
    """Fetches user data from the external API."""
    return {"name": "Alice", "status": "active"}


llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key="not-needed",
    base_url=LLM_URL,
    temperature=0,
)

agent = create_agent(
    llm,
    [fetch_user_profile],
    system_prompt=(
        "Fetch the user profile using the fetch_user_profile tool. "
        "If the tool returns an error, tell the user the specific HTTP "
        "error code and apologize. Do NOT retry — just report the error."
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
@ac.given(error=ac.st.http_errors(probabilities={500: 0.4, 429: 0.4, 200: 0.2}))
@settings(max_examples=5)
def test_agent_handles_api_chaos(error: tuple[int, str]) -> None:
    """The agent must communicate errors clearly, never crash or infinite-loop."""
    status_code, body = error

    with Intercept(resolver=LangChainResolver(lc_tools=[fetch_user_profile])) as ctx:
        if status_code != 200:
            ctx.on("fetch_user_profile").raise_error(
                RuntimeError(f"HTTP {status_code}: {body}")
            )
        else:
            ctx.on("fetch_user_profile").passthrough()

        result = run_agent("Get profile for user 123")

    if status_code != 200:
        # The LLM should mention the error code in its response
        assert str(status_code) in result.output, (
            f"Agent hid the {status_code} error from the user! "
            f"Output: {result.output}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
