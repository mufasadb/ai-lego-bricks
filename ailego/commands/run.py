"""
Workflow execution command for AI Lego Bricks CLI.

This module handles running AI agent workflows from JSON configuration files.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.json import JSON
from rich.table import Table

console = Console()


def run_workflow(
    workflow_file: str,
    input_data: Optional[str] = None,
    output_file: Optional[str] = None,
    verbose: bool = False,
):
    """
    Run an AI agent workflow from a JSON configuration file.

    Args:
        workflow_file: Path to the workflow JSON file
        input_data: Optional input data as JSON string
        output_file: Optional output file path
        verbose: Whether to show detailed execution logs
    """
    workflow_path = Path(workflow_file)

    # Validate workflow file exists
    if not workflow_path.exists():
        console.print(f"[red]Error: Workflow file not found: {workflow_file}[/red]")
        return

    # Parse input data if provided
    parsed_input = {}
    if input_data:
        try:
            parsed_input = json.loads(input_data)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON input data: {e}[/red]")
            return

    # Load workflow configuration
    try:
        with open(workflow_path, "r") as f:
            workflow_config = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid workflow JSON: {e}[/red]")
        return

    if verbose:
        console.print(
            f"[blue]Loading workflow: {workflow_config.get('name', 'Unknown')}[/blue]"
        )
        console.print(
            f"[blue]Description: {workflow_config.get('description', 'No description')}[/blue]"
        )

    # Execute workflow
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=not verbose,
        ) as progress:

            task = progress.add_task("Executing workflow...", total=None)

            # Import and create orchestrator
            from agent_orchestration import AgentOrchestrator

            orchestrator = AgentOrchestrator()

            # Load workflow
            progress.update(task, description="Loading workflow configuration...")
            workflow = orchestrator.load_workflow_from_dict(workflow_config)

            # Execute workflow
            progress.update(task, description="Executing workflow steps...")
            result = orchestrator.execute_workflow(workflow, parsed_input)

            progress.update(task, description="Workflow completed successfully!")

    except Exception as e:
        console.print(f"[red]Error executing workflow: {str(e)}[/red]")
        if verbose:
            import traceback

            console.print(f"[red]Traceback:\n{traceback.format_exc()}[/red]")
        return

    # Display results
    _display_results(result, output_file, verbose)


def _display_results(result: Dict[str, Any], output_file: Optional[str], verbose: bool):
    """Display workflow execution results."""

    if verbose:
        console.print("\n[bold green]Workflow Results:[/bold green]")
        console.print(JSON(json.dumps(result, indent=2)))
    else:
        # Show simplified output
        if "response" in result:
            console.print(f"\n[bold green]Response:[/bold green] {result['response']}")
        elif "result" in result:
            console.print(f"\n[bold green]Result:[/bold green] {result['result']}")
        else:
            console.print(f"\n[bold green]Output:[/bold green] {result}")

    # Save to file if requested
    if output_file:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            console.print(f"[blue]Results saved to: {output_file}[/blue]")
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not save results to file: {e}[/yellow]"
            )

    # Show execution summary
    _show_execution_summary(result)


def _show_execution_summary(result: Dict[str, Any]):
    """Show execution summary with key metrics."""

    # Extract metadata if available
    metadata = result.get("_metadata", {})

    summary_items = []

    if "execution_time" in metadata:
        summary_items.append(f"⏱️  Execution time: {metadata['execution_time']:.2f}s")

    if "steps_executed" in metadata:
        summary_items.append(f"🔄 Steps executed: {metadata['steps_executed']}")

    if "tokens_used" in metadata:
        summary_items.append(f"🔤 Tokens used: {metadata['tokens_used']}")

    if "model_calls" in metadata:
        summary_items.append(f"🤖 Model calls: {metadata['model_calls']}")

    if summary_items:
        summary_text = "\n".join(summary_items)
        console.print(
            Panel(
                summary_text,
                title="[bold blue]Execution Summary[/bold blue]",
                border_style="blue",
            )
        )


def list_available_workflows():
    """List available workflow files in the current project."""

    workflow_dirs = ["agents", "workflows", "."]
    workflows = []

    for directory in workflow_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            for file_path in dir_path.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        config = json.load(f)

                    if "steps" in config:  # Basic validation for workflow
                        workflows.append(
                            {
                                "file": str(file_path),
                                "name": config.get("name", file_path.stem),
                                "description": config.get(
                                    "description", "No description"
                                ),
                            }
                        )
                except Exception:
                    continue  # Skip invalid JSON files

    if workflows:
        console.print("\n[bold blue]Available Workflows:[/bold blue]")

        table = Table()
        table.add_column("File", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description")

        for workflow in workflows:
            table.add_row(workflow["file"], workflow["name"], workflow["description"])

        console.print(table)
    else:
        console.print("[yellow]No workflow files found in current directory[/yellow]")
        console.print(
            "[blue]Try running 'ailego init <project-name>' to create a new project[/blue]"
        )
