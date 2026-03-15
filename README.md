# Agenthesis

Property-based testing framework for AI agents. Agenthesis combines [Hypothesis](https://hypothesis.readthedocs.io/) with tool interception, behavioral invariants, and semantic shrinking to find edge cases in agent behavior.

## Installation

```bash
pip install agenthesis

# For JSON schema validation support
pip install agenthesis[json]
```

## Quick Start

```python
import agenthesis as ac
from agenthesis import Intercept

@ac.max_steps(5)
@ac.never_calls("execute_refund")
@ac.given(ac.st.adversarial_prompts(intensity="high"))
def test_agent_resists_prompt_injection(prompt):
    agent = MyAgent()
    with Intercept(agent) as ctx:
        ctx.on("search").respond({"results": []})
        agent.run(prompt)
```

## Core Concepts

### Tool Interception

`Intercept` is a context manager that wraps agent tool calls, records them, and optionally stubs their behavior:

```python
with Intercept(agent) as ctx:
    ctx.on("search_web").respond({"results": []})      # Fixed response
    ctx.on("database").raise_error(TimeoutError())      # Simulate failure
    ctx.on("api").respond_sequence([{"ok": True}, {}])  # Cycle values

    result = agent.run("find something")

# Inspect the trace after execution
assert ctx.trace.steps <= 3
assert len(ctx.calls) == 2
```

Intercept discovers tools automatically via:
1. Explicit `tools` dict passed to the constructor
2. `ToolKit` protocol (`get_tools()` / `set_tool()`)
3. Attribute discovery (`tool_*` methods on the agent)

### Invariant Decorators

Decorators that assert behavioral properties. They enforce limits at runtime (aborting immediately when exceeded) and also validate post-mortem:

| Decorator | Description |
|-----------|-------------|
| `@max_steps(n)` | Agent completes in at most n steps |
| `@max_llm_calls(n)` | At most n LLM invocations |
| `@max_token_cost(n)` | Total token budget not exceeded |
| `@never_calls(tool)` | Named tool is never invoked |
| `@requires_before(a, b)` | Tool `a` must be called before tool `b` |
| `@output_matches_schema(s)` | Output conforms to a JSON schema |

### Fuzzing Strategies

Hypothesis strategies for generating adversarial inputs:

```python
ac.st.adversarial_prompts(intensity="low"|"medium"|"high")
ac.st.token_overflow(max_tokens=4096)
ac.st.multilingual_prompts()
ac.st.random_prompts(min_size=0, max_size=500)
ac.st.tool_responses(schema=None, error_rate=0.1)
ac.st.malformed_json()
ac.st.http_errors(probabilities=None)
```

### Test Shrinking

After Hypothesis finds a minimal failing byte stream, use `PromptShrinker` to further reduce the prompt to the exact substring that triggers the failure:

```python
from agenthesis import PromptShrinker

shrinker = PromptShrinker(test_fn=lambda p: "DROP" in p)
minimal = shrinker.shrink("Please DROP TABLE users; and also do other things")
# minimal == "DROP"
```

`SequenceShrinker` does the same for lists of tool configurations.

### `ac.given` -- Hypothesis Wrapper

A thin wrapper around `@hypothesis.given` with agent-friendly defaults:
- `max_examples=10` (agents are expensive)
- `deadline=None` (agents are slow)
- Suppresses the `too_slow` health check

```python
@ac.given(ac.st.adversarial_prompts(), max_examples=20)
def test_something(prompt):
    ...
```

## Pytest Integration

Agenthesis registers as a pytest plugin automatically. When an `InvariantViolation` is raised, a Rich-formatted failure report is displayed with trace details (steps, LLM calls, tokens, tool call table).

## Running Tests

```bash
# Run the test suite
pytest tests/ -v

# Run with timeout protection
pytest tests/ -v --timeout=30
```

## Agent Protocol

To get full interception support, agents can implement the `ToolKit` protocol:

```python
from agenthesis import ToolKit

class MyAgent(ToolKit):
    def get_tools(self) -> dict[str, Callable]:
        return {"search": self.search, "calculate": self.calculate}

    def set_tool(self, name: str, fn: Callable) -> None:
        setattr(self, name, fn)
```

Or simply prefix tool methods with `tool_` for automatic discovery:

```python
class MyAgent:
    def tool_search(self, query: str) -> dict:
        ...
```

## License

MIT
