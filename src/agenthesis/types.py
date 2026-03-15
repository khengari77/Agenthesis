"""Core types for Agenthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ToolCall:
    """A single recorded tool invocation."""

    name: str
    arguments: dict[str, Any]
    result: Any
    timestamp: float
    was_intercepted: bool


@dataclass(frozen=True)
class AgentTrace:
    """Complete trace of an agent execution."""

    tool_calls: tuple[ToolCall, ...] = ()
    llm_calls: int = 0
    total_tokens: int = 0
    steps: int = 0
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class AgentResult:
    """The outcome of running an agent."""

    output: Any
    trace: AgentTrace = field(default_factory=AgentTrace)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen=True prevents direct assignment, use object.__setattr__ for validation
        if not isinstance(self.trace, AgentTrace):
            msg = f"trace must be AgentTrace, got {type(self.trace).__name__}"
            raise TypeError(msg)


@runtime_checkable
class ToolKit(Protocol):
    """Protocol for an agent's tool registry."""

    def get_tools(self) -> dict[str, Any]: ...
    def set_tool(self, name: str, fn: Any) -> None: ...


@runtime_checkable
class ToolResolver(Protocol):
    """Protocol for pluggable tool resolution strategies."""

    def resolve(
        self, agent: Any, explicit_tools: dict[str, Callable[..., Any]] | None,
    ) -> dict[str, Callable[..., Any]]:
        """Discover tools from the agent or explicit dict."""
        ...

    def install(
        self,
        agent: Any,
        explicit_tools: dict[str, Callable[..., Any]] | None,
        name: str,
        wrapper: Callable[..., Any],
    ) -> None:
        """Install a wrapped tool on the agent or dict."""
        ...

    def restore(
        self,
        agent: Any,
        explicit_tools: dict[str, Callable[..., Any]] | None,
        name: str,
        original: Callable[..., Any],
    ) -> None:
        """Restore the original tool on the agent or dict."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Minimal protocol that any agent must satisfy to be testable."""

    def run(self, prompt: str) -> AgentResult: ...


class AgenthesisError(Exception):
    """Base exception for Agenthesis."""


class InvariantViolation(AgenthesisError):  # noqa: N818
    """Raised when an agent violates a declared invariant."""

    def __init__(self, invariant: str, message: str, trace: AgentTrace | None = None) -> None:
        self.invariant = invariant
        self.trace = trace
        super().__init__(f"[{invariant}] {message}")


class InterceptError(AgenthesisError):
    """Raised when tool interception fails."""
