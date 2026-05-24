import time
from contextlib import contextmanager
from rich.console import Console

console = Console()

@contextmanager
def status(label):
    """
    A context manager for displaying a status message with a spinner.
    """
    with console.status(f"[bold cyan]{label}..."):
        t = time.perf_counter()
        yield
    console.print(f"  [dim]✓ {label} ({time.perf_counter() - t:.1f}s)[/dim]")


def show_tool_call(name):
    console.print(f"  [yellow]→ {name}[/yellow]")


def show_tool_input(code):
    console.print(f"Input:\n   [dim]{code}[/dim]")


def show_tool_output(output):
    console.print(f"Output:\n   [dim]{output}[/dim]")


def show_answer(text):
    console.print(f"\n[bold green]Answer:[/bold green] {text}")


def show_error(text):
    console.print(f"[bold red]Error:[/bold red] {text}")


def get_user_input(text=""):
    return console.input(f"[bold cyan]You:[/bold cyan] {text}")