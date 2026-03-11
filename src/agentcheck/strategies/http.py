"""Strategies for generating HTTP error responses."""

from __future__ import annotations

from hypothesis import strategies as st

_DEFAULT_PROBABILITIES: dict[int, float] = {
    200: 0.5,
    400: 0.05,
    401: 0.05,
    403: 0.05,
    404: 0.05,
    429: 0.1,
    500: 0.1,
    502: 0.03,
    503: 0.05,
    504: 0.02,
}

_ERROR_BODIES: dict[int, list[str]] = {
    400: ['{"error": "Bad Request"}', '{"message": "Invalid parameters"}', "Bad Request"],
    401: ['{"error": "Unauthorized"}', '{"message": "Invalid API key"}', ""],
    403: ['{"error": "Forbidden"}', '{"message": "Insufficient permissions"}'],
    404: ['{"error": "Not Found"}', '{"message": "Resource does not exist"}', ""],
    429: [
        '{"error": "Rate Limited", "retry_after": 30}',
        '{"error": "Too Many Requests", "retry_after": 60}',
        '{"error": "Rate Limited", "retry_after": 300}',
    ],
    500: [
        '{"error": "Internal Server Error"}',
        "Internal Server Error",
        '{"error": "Unexpected error", "trace_id": "abc123"}',
        "",
    ],
    502: ['{"error": "Bad Gateway"}', "Bad Gateway"],
    503: [
        '{"error": "Service Unavailable"}',
        '{"error": "Service Unavailable", "retry_after": 120}',
        "",
    ],
    504: ['{"error": "Gateway Timeout"}', "Gateway Timeout", ""],
}


@st.composite
def http_errors(
    draw: st.DrawFn,
    probabilities: dict[int, float] | None = None,
) -> tuple[int, str]:
    """Generate HTTP status codes with response bodies.

    Args:
        probabilities: Mapping of status code to probability weight.
                       Defaults to a realistic distribution.

    Returns:
        Tuple of (status_code, response_body).
    """
    probs = probabilities or _DEFAULT_PROBABILITIES

    codes = list(probs.keys())
    weights = list(probs.values())

    # Normalize weights
    total = sum(weights)
    normalized = [w / total for w in weights]

    # Use sampled_from with the weighted distribution
    # Hypothesis doesn't have direct weighted sampling, so we use a float draw
    roll = draw(st.floats(min_value=0.0, max_value=1.0 - 1e-10))
    cumulative = 0.0
    status_code = codes[-1]  # fallback
    for code, weight in zip(codes, normalized, strict=False):
        cumulative += weight
        if roll < cumulative:
            status_code = code
            break

    # Generate body
    if status_code == 200:
        body = draw(
            st.one_of(
                st.just('{"status": "ok"}'),
                st.just('{"data": []}'),
                st.just('{"result": "success"}'),
            )
        )
    elif status_code in _ERROR_BODIES:
        body = draw(st.sampled_from(_ERROR_BODIES[status_code]))
    else:
        body = draw(st.one_of(st.just(""), st.just(f'{{"error": "HTTP {status_code}"}}')))

    return (status_code, body)
