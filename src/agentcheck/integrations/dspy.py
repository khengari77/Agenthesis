"""DSPy synthesis bridge for AgentCheck invariants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class InvariantMetric:
    """Wraps an AgentCheck test function as a DSPy-compatible metric.

    Returns 1.0 if the test passes, 0.0 if it raises any exception.
    Works with duck-typed example/prediction objects — no hard dependency on dspy.
    """

    def __init__(self, test_fn: Callable[..., Any], prompt_field: str = "prompt") -> None:
        self._test_fn = test_fn
        self._prompt_field = prompt_field

    def __call__(self, example: Any, prediction: Any, trace: Any = None) -> float:
        prompt = getattr(example, self._prompt_field, None)
        if prompt is None and hasattr(example, "get"):
            prompt = example.get(self._prompt_field, "")
        try:
            self._test_fn(prompt)
            return 1.0
        except Exception:
            return 0.0
