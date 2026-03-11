"""Strategies for generating tool responses and malformed data."""

from __future__ import annotations

import json

from hypothesis import strategies as st


@st.composite
def tool_responses(
    draw: st.DrawFn,
    schema: dict[str, object] | None = None,
    error_rate: float = 0.1,
) -> dict[str, object]:
    """Generate tool response dicts, with configurable error injection.

    Args:
        schema: Optional JSON-like schema hint for generating plausible responses.
        error_rate: Probability of generating an error response (0.0 to 1.0).
    """
    is_error = draw(st.floats(min_value=0.0, max_value=1.0)) < error_rate

    if is_error:
        return draw(
            st.one_of(
                st.just({}),
                st.just({"error": "Internal Server Error"}),
                st.just({"error": "timeout", "code": 504}),
                st.just({"error": "rate_limited", "retry_after": 60}),
                st.just({"status": "error", "message": None}),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=10),
                    values=st.none(),
                    min_size=1,
                    max_size=3,
                ),
            )
        )

    if schema is not None:
        return draw(_from_schema(schema))

    return draw(
        st.one_of(
            st.fixed_dictionaries(
                {"status": st.just("ok"), "data": st.text(max_size=100)}
            ),
            st.fixed_dictionaries(
                {
                    "results": st.lists(
                        st.dictionaries(
                            keys=st.sampled_from(["id", "name", "value"]),
                            values=st.one_of(st.integers(), st.text(max_size=20)),
                            min_size=1,
                            max_size=3,
                        ),
                        max_size=5,
                    )
                }
            ),
            st.fixed_dictionaries(
                {"count": st.integers(min_value=0, max_value=10000)}
            ),
        )
    )


def _from_schema(schema: dict[str, object]) -> st.SearchStrategy[dict[str, object]]:
    """Generate a dict loosely matching a JSON-schema-like hint."""
    properties = schema.get("properties", {})
    if not properties:
        return st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(st.integers(), st.text(max_size=20), st.none()),
            min_size=0,
            max_size=5,
        )

    fixed: dict[str, st.SearchStrategy[object]] = {}
    for key, prop in properties.items():
        prop_type = prop.get("type", "string") if isinstance(prop, dict) else "string"
        if prop_type == "integer":
            fixed[key] = st.integers(min_value=-1000, max_value=1000)
        elif prop_type == "number":
            fixed[key] = st.floats(min_value=-1000, max_value=1000, allow_nan=False)
        elif prop_type == "boolean":
            fixed[key] = st.booleans()
        elif prop_type == "array":
            fixed[key] = st.lists(st.text(max_size=10), max_size=5)
        else:
            fixed[key] = st.text(max_size=50)

    return st.fixed_dictionaries(fixed)


@st.composite
def malformed_json(draw: st.DrawFn) -> str:
    """Generate strings that look like JSON but are broken."""
    return draw(
        st.one_of(
            # Truncated JSON
            st.just('{"key": "value"'),
            st.just('{"items": [1, 2, 3'),
            st.just('{"nested": {"inner":'),
            # Missing closing braces
            st.just('{"a": 1, "b": 2'),
            # Trailing comma
            st.just('{"a": 1, "b": 2,}'),
            st.just('[1, 2, 3,]'),
            # Duplicate keys
            st.just('{"key": "first", "key": "second"}'),
            # Single quotes instead of double
            st.just("{'key': 'value'}"),
            # Unquoted keys
            st.just("{key: value}"),
            # Control characters in strings
            st.just('{"text": "hello\x00world"}'),
            # NaN/Infinity (invalid JSON)
            st.just('{"value": NaN}'),
            st.just('{"value": Infinity}'),
            # Empty fragments
            st.just(""),
            st.just("null"),
            st.just("undefined"),
            # Random truncation of valid JSON
            st.builds(
                lambda d, n: json.dumps(d)[:n],
                st.fixed_dictionaries(
                    {
                        "data": st.lists(st.integers(), min_size=3, max_size=10),
                        "name": st.text(min_size=5, max_size=20),
                    }
                ),
                st.integers(min_value=1, max_value=30),
            ),
        )
    )
