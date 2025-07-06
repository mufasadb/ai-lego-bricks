"""
Setup verification command for AI Lego Bricks CLI.

This module handles verifying system setup, dependencies, and configuration.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

from ailego.core import get_available_providers

console = Console()


def verify_setup(verbose: bool = False):
    """
    Verify system setup and configuration.
    
    Args:
        verbose: Whether to show detailed verification output
    """
    console.print("\n[bold blue]AI Lego Bricks Setup Verification[/bold blue]")
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        
        # Verification tasks
        total_tasks = 6
        task = progress.add_task("Verifying setup...", total=total_tasks)
        
        results = []
        
        # 1. Check Python version
        progress.update(task, description="Checking Python version...")
        py_result = _check_python_version()
        results.append(("Python Version", py_result[0], py_result[1]))
        progress.advance(task)
        
        # 2. Check dependencies
        progress.update(task, description="Checking dependencies...")
        deps_result = _check_dependencies()
        results.append(("Dependencies", deps_result[0], deps_result[1]))
        progress.advance(task)
        
        # 3. Check environment variables
        progress.update(task, description="Checking environment...")
        env_result = _check_environment()
        results.append(("Environment", env_result[0], env_result[1]))
        progress.advance(task)
        
        # 4. Check providers
        progress.update(task, description="Checking providers...")
        providers_result = _check_providers()
        results.append(("Providers", providers_result[0], providers_result[1]))
        progress.advance(task)
        
        # 5. Check services
        progress.update(task, description="Checking services...")
        services_result = _check_services()
        results.append(("Services", services_result[0], services_result[1]))
        progress.advance(task)
        
        # 6. Check project structure
        progress.update(task, description="Checking project structure...")
        structure_result = _check_project_structure()
        results.append(("Project Structure", structure_result[0], structure_result[1]))
        progress.advance(task)
    
    # Display results
    _display_verification_results(results, verbose)


def _check_python_version() -> Tuple[str, str]:
    """Check Python version compatibility."""
    import sys
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 8):
        return "✓ PASS", f"Python {version_str} (compatible)"
    else:
        return "✗ FAIL", f"Python {version_str} (requires >=3.8)"


def _check_dependencies() -> Tuple[str, str]:
    """Check if required dependencies are installed."""
    required_packages = [
        "pydantic", "httpx", "typer", "rich", "python-dotenv",
        "sentence-transformers", "numpy", "jinja2"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if not missing:
        return "✓ PASS", f"All {len(required_packages)} core dependencies installed"
    else:
        return "⚠ PARTIAL", f"Missing: {', '.join(missing)}"


def _check_environment() -> Tuple[str, str]:
    """Check environment configuration."""
    env_file = Path(".env")
    
    if not env_file.exists():
        return "⚠ WARN", "No .env file found"
    
    # Check for at least one LLM provider
    llm_providers = [
        "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"
    ]
    
    has_llm = any(os.getenv(key) for key in llm_providers)
    
    if has_llm:
        return "✓ PASS", ".env file exists with LLM provider"
    else:
        return "⚠ WARN", ".env file exists but no LLM providers configured"


def _check_providers() -> Tuple[str, str]:
    """Check available service providers."""
    providers = get_available_providers()
    
    total_services = len(providers)
    available_services = sum(1 for p in providers.values() if p)
    
    if available_services >= 2:
        return "✓ PASS", f"{available_services}/{total_services} services available"
    elif available_services >= 1:
        return "⚠ PARTIAL", f"{available_services}/{total_services} services available"
    else:
        return "✗ FAIL", "No services configured"


def _check_services() -> Tuple[str, str]:
    """Check individual service connectivity."""
    issues = []
    
    # Check Supabase
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        try:
            from supabase import create_client
            client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
            # Simple test query
            client.table("test").select("*").limit(1).execute()
        except Exception as e:
            issues.append(f"Supabase: {str(e)[:50]}...")
    
    # Check for local services
    if os.getenv("OLLAMA_URL"):
        try:
            import httpx
            response = httpx.get(f"{os.getenv('OLLAMA_URL')}/api/tags", timeout=5)
            if response.status_code != 200:
                issues.append("Ollama: Not responding")
        except Exception:
            issues.append("Ollama: Connection failed")
    
    if issues:
        return "⚠ WARN", f"Service issues: {'; '.join(issues)}"
    else:
        return "✓ PASS", "All configured services accessible"


def _check_project_structure() -> Tuple[str, str]:
    """Check project structure."""
    current_dir = Path.cwd()
    
    # Check if we're in a proper AI Lego Bricks project
    has_agents = (current_dir / "agents").exists()
    has_config = (current_dir / ".env").exists() or (current_dir / ".env.example").exists()
    
    if has_agents and has_config:
        return "✓ PASS", "Valid AI Lego Bricks project structure"
    elif has_config:
        return "⚠ PARTIAL", "Configuration found, missing agents directory"
    else:
        return "⚠ INFO", "Not in AI Lego Bricks project (run 'ailego init' to create one)"


def _display_verification_results(results: List[Tuple[str, str, str]], verbose: bool):
    """Display verification results in a formatted table."""
    
    table = Table(title="Verification Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")
    
    for component, status, details in results:
        # Color code status
        if status.startswith("✓"):
            status_style = "green"
        elif status.startswith("⚠"):
            status_style = "yellow"
        else:
            status_style = "red"
        
        table.add_row(component, f"[{status_style}]{status}[/{status_style}]", details)
    
    console.print(table)
    
    # Summary
    passed = sum(1 for _, status, _ in results if status.startswith("✓"))
    total = len(results)
    
    if passed == total:
        status_text = "[green]✓ System ready for use![/green]"
    elif passed >= total * 0.7:
        status_text = "[yellow]⚠ System mostly ready, some issues found[/yellow]"
    else:
        status_text = "[red]✗ System needs configuration[/red]"
    
    console.print(f"\n[bold]Summary:[/bold] {status_text}")
    console.print(f"[blue]Passed: {passed}/{total} checks[/blue]")
    
    if verbose:
        _show_detailed_recommendations(results)


def _show_detailed_recommendations(results: List[Tuple[str, str, str]]):
    """Show detailed recommendations for failed checks."""
    recommendations = []
    
    for component, status, details in results:
        if status.startswith("✗") or status.startswith("⚠"):
            if "Python" in component:
                recommendations.append("• Upgrade to Python 3.8 or higher")
            elif "Dependencies" in component:
                recommendations.append("• Run: pip install -r requirements.txt")
            elif "Environment" in component:
                recommendations.append("• Copy .env.example to .env and add your API keys")
            elif "Providers" in component:
                recommendations.append("• Add API keys for at least one LLM provider")
            elif "Services" in component:
                recommendations.append("• Check service connectivity and credentials")
            elif "Project" in component:
                recommendations.append("• Run: ailego init <project-name> to create a project")
    
    if recommendations:
        recommendations_text = "\n".join(recommendations)
        console.print(Panel(
            recommendations_text,
            title="[bold red]Recommendations[/bold red]",
            border_style="red"
        ))