"""Tests for the DSPy synthesis bridge."""

from __future__ import annotations

from agentcheck.integrations.dspy import InvariantMetric


class _Example:
    """Duck-typed example object (no dspy dependency needed)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestInvariantMetric:
    def test_metric_returns_1_on_pass(self) -> None:
        def passing_test(prompt):
            pass  # no exception = pass

        metric = InvariantMetric(passing_test)
        example = _Example(prompt="hello")
        assert metric(example, prediction=None) == 1.0

    def test_metric_returns_0_on_violation(self) -> None:
        def failing_test(prompt):
            raise ValueError("invariant violated")

        metric = InvariantMetric(failing_test)
        example = _Example(prompt="hello")
        assert metric(example, prediction=None) == 0.0

    def test_metric_custom_prompt_field(self) -> None:
        def passing_test(prompt):
            pass

        metric = InvariantMetric(passing_test, prompt_field="input_text")
        example = _Example(input_text="hello")
        assert metric(example, prediction=None) == 1.0

    def test_metric_dict_like_example(self) -> None:
        def passing_test(prompt):
            pass

        metric = InvariantMetric(passing_test)
        example = {"prompt": "hello"}
        assert metric(example, prediction=None) == 1.0

    def test_metric_missing_prompt_field(self) -> None:
        def test_fn(prompt):
            if prompt is None:
                raise ValueError("no prompt")

        metric = InvariantMetric(test_fn)
        example = _Example(other_field="hello")
        assert metric(example, prediction=None) == 0.0
