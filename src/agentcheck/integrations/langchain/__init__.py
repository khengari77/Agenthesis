"""LangChain integration for AgentCheck."""

from __future__ import annotations

try:
    from langchain_core.callbacks import BaseCallbackHandler as _  # noqa: F401
except ImportError as e:
    msg = (
        "langchain-core is required for the LangChain integration. "
        "Install with: pip install agentcheck[langchain]"
    )
    raise ImportError(msg) from e

from agentcheck.integrations.langchain.adapter import LangChainAgentAdapter
from agentcheck.integrations.langchain.callback import AgentCheckCallbackHandler
from agentcheck.integrations.langchain.resolver import LangChainResolver

__all__ = [
    "AgentCheckCallbackHandler",
    "LangChainAgentAdapter",
    "LangChainResolver",
]
