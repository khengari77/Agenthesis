"""AgentCheck: Property-based testing for AI agents."""

__version__ = "0.1.0"

from agentcheck import strategies as st
from agentcheck._given import given
from agentcheck.intercept import Intercept
from agentcheck.properties import (
    max_llm_calls,
    max_steps,
    max_token_cost,
    never_calls,
    output_matches_schema,
    requires_before,
)
from agentcheck.types import (
    AgentCheckError,
    AgentProtocol,
    AgentResult,
    AgentTrace,
    InterceptError,
    InvariantViolation,
    ToolCall,
    ToolKit,
)

__all__ = [
    "AgentCheckError",
    "AgentProtocol",
    "AgentResult",
    "AgentTrace",
    "Intercept",
    "InterceptError",
    "InvariantViolation",
    "ToolCall",
    "ToolKit",
    "given",
    "max_llm_calls",
    "max_steps",
    "max_token_cost",
    "never_calls",
    "output_matches_schema",
    "requires_before",
    "st",
]
