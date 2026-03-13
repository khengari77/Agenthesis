"""AgentCheck: Property-based testing for AI agents."""

__version__ = "0.1.0"

from agentcheck import strategies as st
from agentcheck._given import given
from agentcheck.intercept import Intercept
from agentcheck.properties import (
    lark_grammar,
    max_llm_calls,
    max_steps,
    max_token_cost,
    never_calls,
    output_matches_grammar,
    output_matches_schema,
    requires_before,
)
from agentcheck.shrink import PromptShrinker, SequenceShrinker
from agentcheck.types import (
    AgentCheckError,
    AgentProtocol,
    AgentResult,
    AgentTrace,
    InterceptError,
    InvariantViolation,
    ToolCall,
    ToolKit,
    ToolResolver,
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
    "ToolResolver",
    "given",
    "lark_grammar",
    "PromptShrinker",
    "SequenceShrinker",
    "max_llm_calls",
    "max_steps",
    "max_token_cost",
    "never_calls",
    "output_matches_grammar",
    "output_matches_schema",
    "requires_before",
    "st",
]
