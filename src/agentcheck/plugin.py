"""Pytest plugin for AgentCheck — wires up Rich reporting on test outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentcheck.report import report_failure, report_success
from agentcheck.types import InvariantViolation

if TYPE_CHECKING:
    import pytest


def pytest_exception_interact(
    node: pytest.Item,
    call: pytest.CallInfo[Any],
    report: pytest.TestReport,
) -> None:
    """Display a Rich failure report when an InvariantViolation is raised."""
    if call.excinfo is None:
        return
    if not call.excinfo.errisinstance(InvariantViolation):
        return

    violation: InvariantViolation = call.excinfo.value
    test_name = node.name
    report_failure(test_name, violation)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Display a Rich success summary for passing tests."""
    if report.when != "call":
        return
    if report.passed:
        report_success(report.nodeid, examples_run=0)
