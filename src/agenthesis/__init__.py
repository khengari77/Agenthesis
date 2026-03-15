"""Agenthesis: Property-based testing for AI agents."""

__version__ = "0.1.0"

from agenthesis import strategies as st
from agenthesis._given import given
from agenthesis.intercept import DefaultResolver, Intercept
from agenthesis.properties import (
    lark_grammar,
    max_llm_calls,
    max_steps,
    max_token_cost,
    never_calls,
    output_matches_grammar,
    output_matches_schema,
    requires_before,
)
from agenthesis.shrink import PromptShrinker, SequenceShrinker
from agenthesis.types import (
    AgenthesisError,
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
    "AgenthesisError",
    "AgentProtocol",
    "DefaultResolver",
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
