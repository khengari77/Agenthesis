"""LangChain integration for Agenthesis."""

from __future__ import annotations

try:
    from langchain_core.callbacks import BaseCallbackHandler as _  # noqa: F401
except ImportError as e:
    msg = (
        "langchain-core is required for the LangChain integration. "
        "Install with: pip install agenthesis[langchain]"
    )
    raise ImportError(msg) from e

from agenthesis.integrations.langchain.adapter import LangChainAgentAdapter
from agenthesis.integrations.langchain.callback import AgenthesisCallbackHandler
from agenthesis.integrations.langchain.resolver import LangChainResolver

__all__ = [
    "AgenthesisCallbackHandler",
    "LangChainAgentAdapter",
    "LangChainResolver",
]
