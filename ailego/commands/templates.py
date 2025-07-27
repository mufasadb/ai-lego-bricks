"""
Templates listing command for AI Lego Bricks CLI.

This module handles listing and displaying available project and agent templates.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

from ailego.commands.create import AGENT_TEMPLATES

console = Console()


def list_all_templates():
    """List all available templates for projects and agents."""

    console.print("\n[bold blue]🧱 AI Lego Bricks Templates[/bold blue]")

    # Project templates
    _show_project_templates()

    # Agent templates
    _show_agent_templates()

    # Usage examples
    _show_usage_examples()


def _show_project_templates():
    """Show available project templates."""

    console.print("\n[bold green]📁 Project Templates[/bold green]")

    project_templates = [
        {
            "name": "basic",
            "description": "Simple chat and text processing agents",
            "features": "Core LLM, Basic agents, Simple configuration",
        },
        {
            "name": "advanced",
            "description": "Full-featured setup with memory and multi-modal",
            "features": "Memory services, Vision, TTS, Advanced orchestration",
        },
        {
            "name": "research",
            "description": "Specialized for document analysis and research",
            "features": "Document processing, Knowledge extraction, Research workflows",
        },
    ]

    table = Table()
    table.add_column("Template", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Features")

    for template in project_templates:
        table.add_row(template["name"], template["description"], template["features"])

    console.print(table)
    console.print(
        "[blue]Usage: ailego init <project-name> --template <template-name>[/blue]"
    )


def _show_agent_templates():
    """Show available agent templates."""

    console.print("\n[bold green]🤖 Agent Templates[/bold green]")

    table = Table()
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Features")

    for agent_type, template_info in AGENT_TEMPLATES.items():
        # Extract key features from template
        features = _extract_agent_features(template_info["template"])

        table.add_row(
            agent_type, template_info["name"], template_info["description"], features
        )

    console.print(table)
    console.print("[blue]Usage: ailego create <type> --name <agent-name>[/blue]")


def _extract_agent_features(template):
    """Extract key features from an agent template."""
    features = []

    # Check for different step types
    step_types = {step.get("type") for step in template.get("steps", [])}

    if "llm_chat" in step_types:
        features.append("LLM Chat")
    if "llm_vision" in step_types:
        features.append("Vision")
    if "memory_store" in step_types or "memory_retrieve" in step_types:
        features.append("Memory")
    if "tts" in step_types:
        features.append("TTS")
    if "document_processing" in step_types:
        features.append("Documents")
    if "llm_structured" in step_types:
        features.append("Structured Output")

    # Check for streaming
    for step in template.get("steps", []):
        if step.get("config", {}).get("stream"):
            features.append("Streaming")
            break

    return ", ".join(features) if features else "Basic"


def _show_usage_examples():
    """Show usage examples for templates."""

    console.print("\n[bold green]💡 Usage Examples[/bold green]")

    examples = [
        {
            "title": "Start a New Project",
            "commands": [
                "ailego init my-ai-project",
                "cd my-ai-project",
                "ailego verify",
            ],
        },
        {
            "title": "Create Custom Agents",
            "commands": [
                "ailego create chat --name customer-support",
                "ailego create document-analysis --name pdf-analyzer",
                "ailego create research --name market-research",
            ],
        },
        {
            "title": "Run Agent Workflows",
            "commands": [
                "ailego run agents/customer-support.json",
                "ailego run agents/pdf-analyzer.json --verbose",
                "ailego run agents/market-research.json --output results.json",
            ],
        },
    ]

    panels = []
    for example in examples:
        command_text = "\n".join(f"[blue]{cmd}[/blue]" for cmd in example["commands"])
        panels.append(Panel(command_text, title=example["title"], border_style="green"))

    console.print(Columns(panels, equal=True))


def show_template_details(template_type: str):
    """Show detailed information about a specific template."""

    if template_type in AGENT_TEMPLATES:
        template_info = AGENT_TEMPLATES[template_type]

        console.print(f"\n[bold blue]{template_info['name']}[/bold blue]")
        console.print(f"[green]{template_info['description']}[/green]")

        # Show workflow steps
        console.print("\n[bold yellow]Workflow Steps:[/bold yellow]")

        template = template_info["template"]
        for i, step in enumerate(template.get("steps", []), 1):
            step_type = step.get("type", "unknown")
            step_id = step.get("id", f"step_{i}")
            console.print(f"[cyan]{i}. {step_id}[/cyan] - {step_type}")

        # Show configuration
        config = template.get("config", {})
        if config:
            console.print("\n[bold yellow]Configuration:[/bold yellow]")
            for key, value in config.items():
                console.print(f"[green]• {key}:[/green] {value}")

        console.print(
            f"\n[blue]Create with: ailego create {template_type} --name <your-name>[/blue]"
        )

    else:
        console.print(f"[red]Unknown template type: {template_type}[/red]")
        console.print(
            "[blue]Use 'ailego list-templates' to see available templates[/blue]"
        )


def show_project_template_details(template_name: str):
    """Show detailed information about a project template."""

    project_details = {
        "basic": {
            "description": "A minimal setup for simple AI agents",
            "includes": [
                "Basic chat agent",
                "Simple text processing",
                "Core LLM integration",
                "Environment configuration",
            ],
            "use_cases": [
                "Learning AI agent development",
                "Simple chatbots",
                "Basic text processing tasks",
            ],
        },
        "advanced": {
            "description": "Full-featured AI agent development environment",
            "includes": [
                "Memory-enabled agents",
                "Multi-modal processing (vision, audio)",
                "Text-to-speech integration",
                "Advanced orchestration",
                "Structured output generation",
            ],
            "use_cases": [
                "Complex AI applications",
                "Multi-modal processing",
                "Research and analysis tools",
                "Production AI systems",
            ],
        },
        "research": {
            "description": "Specialized for document analysis and research workflows",
            "includes": [
                "Document processing agents",
                "Knowledge extraction",
                "Research synthesis",
                "Multi-document analysis",
                "Report generation",
            ],
            "use_cases": [
                "Academic research",
                "Document analysis",
                "Content synthesis",
                "Knowledge management",
            ],
        },
    }

    if template_name in project_details:
        details = project_details[template_name]

        console.print(f"\n[bold blue]Project Template: {template_name}[/bold blue]")
        console.print(f"[green]{details['description']}[/green]")

        console.print("\n[bold yellow]Includes:[/bold yellow]")
        for item in details["includes"]:
            console.print(f"[cyan]• {item}[/cyan]")

        console.print("\n[bold yellow]Use Cases:[/bold yellow]")
        for use_case in details["use_cases"]:
            console.print(f"[green]• {use_case}[/green]")

        console.print(
            f"\n[blue]Create with: ailego init <project-name> --template {template_name}[/blue]"
        )

    else:
        console.print(f"[red]Unknown project template: {template_name}[/red]")
        console.print("[blue]Available templates: basic, advanced, research[/blue]")
