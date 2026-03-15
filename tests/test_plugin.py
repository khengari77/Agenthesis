"""Tests for the pytest plugin auto-shrinking functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

from agenthesis.plugin import _extract_failing_prompt


class TestExtractFailingPrompt:
    def test_parses_hypothesis_msg(self) -> None:
        exc = Exception("Falsifying example: test_func(prompt='hello world')")
        excinfo = MagicMock()
        excinfo.value = exc
        excinfo.errisinstance.return_value = False

        result = _extract_failing_prompt(excinfo)
        assert result == "hello world"

    def test_parses_double_quoted_prompt(self) -> None:
        exc = Exception('Falsifying example: test_func(prompt="hello world")')
        excinfo = MagicMock()
        excinfo.value = exc
        excinfo.errisinstance.return_value = False

        result = _extract_failing_prompt(excinfo)
        assert result == "hello world"

    def test_returns_none_for_non_hypothesis(self) -> None:
        exc = Exception("some other error")
        excinfo = MagicMock()
        excinfo.value = exc
        excinfo.errisinstance.return_value = False

        result = _extract_failing_prompt(excinfo)
        assert result is None

    def test_walks_exception_chain(self) -> None:
        inner = Exception("Falsifying example: func(prompt='nested')")
        outer = Exception("wrapper")
        outer.__cause__ = inner

        excinfo = MagicMock()
        excinfo.value = outer
        excinfo.errisinstance.return_value = False

        result = _extract_failing_prompt(excinfo)
        assert result == "nested"

    def test_no_infinite_loop_on_cycle(self) -> None:
        exc = Exception("no match")
        exc.__cause__ = exc  # cycle

        excinfo = MagicMock()
        excinfo.value = exc
        excinfo.errisinstance.return_value = False

        result = _extract_failing_prompt(excinfo)
        assert result is None


class TestAutoShrinkIntegration:
    def test_auto_shrink_produces_shorter_input(self) -> None:
        """Verify auto-shrinking produces a result no longer than the original."""
        from agenthesis.shrink import PromptShrinker

        def test_fn(p: str) -> bool:
            return "fail" in p

        shrinker = PromptShrinker(test_fn, max_iterations=20)
        result = shrinker.shrink("this will fail because of the keyword")

        assert "fail" in result
        assert len(result) <= len("this will fail because of the keyword")
