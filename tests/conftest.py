"""Shared fixtures for Agenthesis tests."""

from __future__ import annotations

import pytest

from agenthesis._testing import DummyAgent


@pytest.fixture
def agent() -> DummyAgent:
    """A basic dummy agent with default tools."""
    return DummyAgent()


@pytest.fixture
def bad_agent() -> DummyAgent:
    """A dummy agent that calls forbidden tools."""
    return DummyAgent(failure_mode="wrong_tool")


@pytest.fixture
def looping_agent() -> DummyAgent:
    """A dummy agent that loops excessively."""
    return DummyAgent(failure_mode="infinite_loop")


@pytest.fixture
def expensive_agent() -> DummyAgent:
    """A dummy agent that uses excessive tokens."""
    return DummyAgent(failure_mode="token_explosion")
