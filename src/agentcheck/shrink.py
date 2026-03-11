"""Post-Hypothesis secondary shrinker for agent-specific artifacts."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class PromptShrinker:
    """Binary-search shrinker for prompts.

    After Hypothesis finds a minimal failing byte stream, this shrinker
    further reduces the prompt text to find the exact substring that
    triggers the failure.
    """

    def __init__(
        self,
        test_fn: Callable[[str], bool],
        max_iterations: int = 50,
    ) -> None:
        """Initialize the shrinker.

        Args:
            test_fn: A function that returns True if the prompt causes a failure.
            max_iterations: Maximum shrinking iterations to prevent infinite loops.
        """
        self._test_fn = test_fn
        self._max_iterations = max_iterations
        self._history: list[tuple[str, bool]] = []

    @property
    def history(self) -> list[tuple[str, bool]]:
        """Get the shrinking history as (prompt, failed) pairs."""
        return list(self._history)

    def shrink(self, failing_prompt: str) -> str:
        """Shrink a failing prompt to its minimal failing substring.

        Strategy:
        1. Try halves (binary search on text)
        2. Try removing each sentence
        3. Try removing each word
        """
        current = failing_prompt
        iterations = 0

        # Phase 1: Binary halving
        current = self._shrink_by_halving(current, iterations)
        iterations += len(self._history)

        # Phase 2: Sentence removal
        if iterations < self._max_iterations:
            current = self._shrink_by_sentences(current, iterations)
            iterations = len(self._history)

        # Phase 3: Word removal
        if iterations < self._max_iterations:
            current = self._shrink_by_words(current, iterations)

        return current

    def _still_fails(self, prompt: str) -> bool:
        """Test if a prompt still causes failure, recording the attempt."""
        result = self._test_fn(prompt)
        self._history.append((prompt, result))
        return result

    def _shrink_by_halving(self, prompt: str, iterations: int) -> str:
        """Try splitting the prompt in half and testing each half."""
        if len(prompt) <= 1 or iterations >= self._max_iterations:
            return prompt

        mid = len(prompt) // 2
        first_half = prompt[:mid]
        second_half = prompt[mid:]

        if self._still_fails(first_half):
            return self._shrink_by_halving(first_half, iterations + 1)
        if self._still_fails(second_half):
            return self._shrink_by_halving(second_half, iterations + 1)

        return prompt

    def _shrink_by_sentences(self, prompt: str, iterations: int) -> str:
        """Try removing each sentence to see if failure persists."""
        sentences = re.split(r'(?<=[.!?])\s+', prompt)
        if len(sentences) <= 1:
            return prompt

        changed = True
        while changed and iterations < self._max_iterations:
            changed = False
            for i in range(len(sentences)):
                reduced = sentences[:i] + sentences[i + 1 :]
                candidate = " ".join(reduced)
                iterations += 1
                if candidate and self._still_fails(candidate):
                    sentences = reduced
                    changed = True
                    break

        return " ".join(sentences)

    def _shrink_by_words(self, prompt: str, iterations: int) -> str:
        """Try removing each word to see if failure persists."""
        words = prompt.split()
        if len(words) <= 1:
            return prompt

        changed = True
        while changed and iterations < self._max_iterations:
            changed = False
            for i in range(len(words)):
                reduced = words[:i] + words[i + 1 :]
                candidate = " ".join(reduced)
                iterations += 1
                if candidate and self._still_fails(candidate):
                    words = reduced
                    changed = True
                    break

        return " ".join(words)


class SequenceShrinker:
    """Shrinks a sequence of tool call configurations to find the minimal
    set that triggers a failure."""

    def __init__(
        self,
        test_fn: Callable[[list[object]], bool],
        max_iterations: int = 50,
    ) -> None:
        self._test_fn = test_fn
        self._max_iterations = max_iterations

    def shrink(self, failing_sequence: list[object]) -> list[object]:
        """Remove items from the sequence while failure persists."""
        current = list(failing_sequence)
        iterations = 0

        changed = True
        while changed and iterations < self._max_iterations:
            changed = False
            for i in range(len(current)):
                candidate = current[:i] + current[i + 1 :]
                iterations += 1
                if candidate and self._test_fn(candidate):
                    current = candidate
                    changed = True
                    break

        return current
