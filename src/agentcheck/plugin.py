"""Pytest plugin for AgentCheck — wires up Rich reporting on test outcomes."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agentcheck.report import report_failure, report_success
from agentcheck.shrink import PromptShrinker
from agentcheck.types import InvariantViolation

if TYPE_CHECKING:
    import pytest

_FALSIFYING_RE = re.compile(r"Falsifying example:.*?\(.*?prompt=['\"](.+?)['\"]", re.DOTALL)


def _extract_failing_prompt(excinfo: pytest.ExceptionInfo[Any]) -> str | None:
    """Extract the failing prompt from a Hypothesis exception chain."""
    exc: BaseException | None = excinfo.value
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        match = _FALSIFYING_RE.search(str(exc))
        if match:
            return match.group(1)
        exc = exc.__cause__ or exc.__context__
    return None


def _find_invariant_violation(excinfo: pytest.ExceptionInfo[Any]) -> InvariantViolation | None:
    """Walk the exception chain to find an InvariantViolation.

    Hypothesis may wrap the original error in its own exception types
    (e.g., Flaky, FalsifyingExample), so we traverse __cause__ and __context__.
    """
    if excinfo.errisinstance(InvariantViolation):
        return excinfo.value

    exc: BaseException | None = excinfo.value
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, InvariantViolation):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def pytest_exception_interact(
    node: pytest.Item,
    call: pytest.CallInfo[Any],
    report: pytest.TestReport,
) -> None:
    """Display a Rich failure report when an InvariantViolation is raised."""
    if call.excinfo is None:
        return

    violation = _find_invariant_violation(call.excinfo)
    if violation is None:
        return

    shrunk_input = None
    prompt = _extract_failing_prompt(call.excinfo)
    if prompt is not None and hasattr(node, "obj") and callable(node.obj):
        try:

            def test_fn(p: str) -> bool:
                try:
                    node.obj(p)
                except Exception:
                    return True
                return False

            shrunk_input = PromptShrinker(test_fn, max_iterations=20).shrink(prompt)
        except Exception:
            pass  # best-effort

    report_failure(node.name, violation, shrunk_input=shrunk_input)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Display a Rich success summary for passing tests."""
    if report.when != "call":
        return
    if report.passed:
        report_success(report.nodeid, examples_run=0)
