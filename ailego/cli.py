"""
AI Lego Bricks CLI - Command-line interface for managing AI agent projects.

This module provides the main CLI interface with commands for initializing projects,
verifying setups, running workflows, and creating new agents.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from ailego.core import get_version, get_available_providers

# Import commands will be done locally to avoid redefinition issues

app = typer.Typer(
    name="ailego",
    help="AI Lego Bricks - Modular LLM Agent System",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Show the current version of AI Lego Bricks."""
    console.print(f"AI Lego Bricks v{get_version()}", style="bold green")


@app.command()
def status():
    """Show system status and available providers."""
    providers = get_available_providers()

    console.print("\n[bold blue]AI Lego Bricks System Status[/bold blue]")
    console.print(f"Version: {get_version()}")

    # Create provider status table
    table = Table(title="Available Providers")
    table.add_column("Service", style="cyan")
    table.add_column("Providers", style="green")
    table.add_column("Status", style="yellow")

    for service, provider_list in providers.items():
        status_text = "✓ Ready" if provider_list else "⚠ Not configured"
        provider_text = ", ".join(provider_list) if provider_list else "None"
        table.add_row(service.upper(), provider_text, status_text)

    console.print(table)


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project to initialize"),
    template: str = typer.Option(
        "basic", help="Template to use (basic, advanced, research)"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing project"),
):
    """Initialize a new AI Lego Bricks project."""
    from ailego.commands.init import init_project

    init_project(project_name, template, force)


@app.command()
def verify(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed verification"
    ),
):
    """Verify system setup and configuration."""
    from ailego.commands.verify import verify_setup

    verify_setup(verbose)


@app.command()
def run(
    workflow_file: str = typer.Argument(..., help="Path to workflow JSON file"),
    input_data: Optional[str] = typer.Option(None, help="Input data as JSON string"),
    output_file: Optional[str] = typer.Option(None, help="Output file path"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed execution"
    ),
):
    """Run an AI agent workflow."""
    from ailego.commands.run import run_workflow

    run_workflow(workflow_file, input_data, output_file, verbose)


@app.command()
def create(
    agent_type: str = typer.Argument(..., help="Type of agent to create"),
    name: str = typer.Option(None, help="Name for the new agent"),
    interactive: bool = typer.Option(True, help="Interactive agent creation"),
):
    """Create a new AI agent configuration."""
    from ailego.commands.create import create_agent

    create_agent(agent_type, name, interactive)


@app.command()
def list_templates():
    """List available project and agent templates."""
    from ailego.commands.templates import list_all_templates

    list_all_templates()


@app.command()
def examples():
    """Show example workflows and usage patterns."""
    console.print("\n[bold blue]AI Lego Bricks Examples[/bold blue]")

    examples_text = """
[bold green]Quick Start Examples:[/bold green]

1. [yellow]Initialize a new project:[/yellow]
   ailego init my-ai-project

2. [yellow]Verify your setup:[/yellow]
   ailego verify --verbose

3. [yellow]Run a simple chat agent:[/yellow]
   ailego run examples/simple_chat_agent.json

4. [yellow]Create a new document analysis agent:[/yellow]
   ailego create document-analysis --name "doc-analyzer"

5. [yellow]Check system status:[/yellow]
   ailego status

[bold green]Advanced Examples:[/bold green]

• Research Agent: Analyze multiple documents and generate insights
• Vision Agent: Process images and generate descriptions
• Streaming Agent: Real-time conversation with TTS output
• Memory Agent: Store and retrieve contextual information

Use 'ailego create --help' to see all available agent types.
"""

    console.print(Panel(examples_text, title="Examples", border_style="blue"))


# Test Management Commands
test_app = typer.Typer(name="test", help="Manage VCR cassettes and run tests")
app.add_typer(test_app, name="test")


@test_app.command("record")
def test_record(
    service: Optional[str] = typer.Argument(
        None, help="Service to record (chat, http, memory, etc.)"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing cassettes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Record VCR cassettes by running integration tests with real API calls."""
    from ailego.commands.test import record_cassettes

    record_cassettes(service, force, verbose)


@test_app.command("update")
def test_update(
    service: Optional[str] = typer.Argument(None, help="Service to update"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Update existing cassettes with new episodes."""
    from ailego.commands.test import update_cassettes

    update_cassettes(service, verbose)


@test_app.command("unit")
def test_unit(
    service: Optional[str] = typer.Argument(None, help="Service to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    coverage: bool = typer.Option(False, "--coverage", help="Generate coverage report"),
):
    """Run unit tests using recorded cassettes (no network calls)."""
    from ailego.commands.test import run_unit_tests

    run_unit_tests(service, verbose, coverage)


@test_app.command("integration")
def test_integration(
    service: Optional[str] = typer.Argument(None, help="Service to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Run integration tests with real API calls."""
    from ailego.commands.test import run_integration_tests

    run_integration_tests(service, verbose)


@test_app.command("run")
def test_run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    coverage: bool = typer.Option(False, "--coverage", help="Generate coverage report"),
):
    """Run all tests (unit + integration)."""
    from ailego.commands.test import run_all_tests

    run_all_tests(verbose, coverage)


@test_app.command("clean-cassettes")
def test_clean(
    service: Optional[str] = typer.Argument(None, help="Service to clean"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clean old or invalid VCR cassettes."""
    from ailego.commands.test import clean_cassettes

    clean_cassettes(service, yes)


@test_app.command("info")
def test_info():
    """Show information about existing VCR cassettes."""
    from ailego.commands.test import show_cassette_info

    show_cassette_info()


@test_app.command("validate")
def test_validate():
    """Validate that cassettes don't contain sensitive data."""
    from ailego.commands.test import validate_cassettes

    validate_cassettes()


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    main()
