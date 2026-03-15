"""Rich terminal reporting for Agenthesis test results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from agenthesis.types import InvariantViolation

console = Console(stderr=True, force_terminal=True)


def report_failure(
    test_name: str,
    violation: InvariantViolation,
    shrunk_input: Any | None = None,
) -> None:
    """Display a rich panel with failure details."""
    trace = violation.trace

    # Header
    title = Text(f" INVARIANT VIOLATION: {violation.invariant} ", style="bold white on red")

    # Build content
    lines: list[str] = [
        f"[bold]Test:[/bold] {test_name}",
        f"[bold]Violation:[/bold] {violation}",
    ]

    if shrunk_input is not None:
        input_str = repr(shrunk_input)
        if len(input_str) > 200:
            input_str = input_str[:200] + "..."
        lines.append(f"[bold]Minimal failing input:[/bold] {input_str}")

    if trace is not None:
        lines.append("")
        lines.append(f"[bold]Steps:[/bold] {trace.steps}")
        lines.append(f"[bold]LLM calls:[/bold] {trace.llm_calls}")
        lines.append(f"[bold]Total tokens:[/bold] {trace.total_tokens}")
        lines.append(f"[bold]Duration:[/bold] {trace.duration_seconds:.3f}s")

        if trace.tool_calls:
            lines.append("")
            lines.append("[bold]Tool calls:[/bold]")
            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
            table.add_column("#", style="dim", width=3)
            table.add_column("Tool", style="green")
            table.add_column("Arguments", max_width=40)
            table.add_column("Intercepted", style="yellow")
            table.add_column("Result", max_width=60)

            for i, tc in enumerate(trace.tool_calls, 1):
                args_str = repr(tc.arguments)
                if len(args_str) > 40:
                    args_str = args_str[:37] + "..."
                result_str = repr(tc.result)
                if len(result_str) > 60:
                    result_str = result_str[:57] + "..."
                table.add_row(
                    str(i),
                    tc.name,
                    args_str,
                    "yes" if tc.was_intercepted else "no",
                    result_str,
                )

            content = "\n".join(lines)
            panel = Panel(content, title=title, border_style="red")
            console.print(panel)
            console.print(table)
            return

    content = "\n".join(lines)
    panel = Panel(content, title=title, border_style="red")
    console.print(panel)


def report_success(test_name: str, examples_run: int) -> None:
    """Display a brief success summary."""
    console.print(
        f"[bold green] PASSED [/bold green] {test_name} "
        f"[dim]({examples_run} examples)[/dim]"
    )


def report_shrinking(iteration: int, prompt_length: int, still_fails: bool) -> None:
    """Display shrinking progress."""
    status = "[red]fails[/red]" if still_fails else "[green]passes[/green]"
    console.print(
        f"  [dim]shrink #{iteration}:[/dim] {prompt_length} chars -> {status}"
    )
