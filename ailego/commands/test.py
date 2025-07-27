"""
Test management commands for AI Lego Bricks.

This module provides commands for managing VCR cassettes, running tests,
and maintaining the testing infrastructure.
"""

import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


console = Console()


def get_project_root() -> Path:
    """Find the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_tests_dir() -> Path:
    """Get the tests directory path."""
    return get_project_root() / "tests"


def get_cassettes_dir() -> Path:
    """Get the cassettes directory path."""
    return get_tests_dir() / "cassettes"


def run_pytest_command(command: List[str], description: str) -> bool:
    """
    Run a pytest command with progress indication.
    
    Args:
        command: List of command arguments
        description: Description for progress display
        
    Returns:
        True if command succeeded, False otherwise
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)
        
        try:
            result = subprocess.run(
                command,
                cwd=get_project_root(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            progress.update(task, completed=True)
            
            if result.returncode != 0:
                console.print(f"[red]Command failed with exit code {result.returncode}[/red]")
                console.print(f"[red]Error output:[/red]\n{result.stderr}")
                console.print(f"[red]Standard output:[/red]\n{result.stdout}")
                return False
            
            console.print(f"[green]✓ {description} completed successfully[/green]")
            return True
            
        except subprocess.TimeoutExpired:
            progress.update(task, completed=True)
            console.print(f"[red]✗ {description} timed out[/red]")
            return False
        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"[red]✗ {description} failed: {str(e)}[/red]")
            return False


def record_cassettes(
    service: Optional[str] = None,
    force: bool = False,
    verbose: bool = False
) -> None:
    """
    Record VCR cassettes by running integration tests.
    
    Args:
        service: Specific service to record (chat, http, memory, etc.)
        force: Whether to overwrite existing cassettes
        verbose: Whether to show detailed output
    """
    console.print("\n[bold blue]🎬 Recording VCR Cassettes[/bold blue]")
    
    # Check if tests directory exists
    tests_dir = get_tests_dir()
    if not tests_dir.exists():
        console.print("[red]✗ Tests directory not found. Run from project root.[/red]")
        raise typer.Exit(1)
    
    # Build pytest command
    command = ["python", "-m", "pytest"]
    
    if service:
        # Test specific service
        integration_test = tests_dir / "integration" / f"test_{service}_integration.py"
        if not integration_test.exists():
            console.print(f"[red]✗ Integration tests for '{service}' not found[/red]")
            raise typer.Exit(1)
        command.append(str(integration_test))
    else:
        # Test all integration tests
        command.append(str(tests_dir / "integration"))
    
    # Set recording mode
    record_mode = "rewrite" if force else "once"
    command.extend(["--record-mode", record_mode])
    
    # Add verbosity
    if verbose:
        command.append("-v")
    
    # Add markers
    command.extend(["-m", "integration"])
    
    console.print(f"Recording mode: [yellow]{record_mode}[/yellow]")
    if service:
        console.print(f"Service: [yellow]{service}[/yellow]")
    else:
        console.print("Recording: [yellow]All services[/yellow]")
    
    # Check for required environment variables
    env_warnings = []
    required_vars = [
        ("OPENAI_API_KEY", "OpenAI integration tests"),
        ("ANTHROPIC_API_KEY", "Anthropic integration tests"),
        ("GOOGLE_AI_STUDIO_KEY", "Gemini integration tests"),
        ("SUPABASE_URL", "Supabase memory tests"),
        ("NEO4J_URI", "Neo4j memory tests"),
    ]
    
    for var_name, description in required_vars:
        if not os.getenv(var_name):
            env_warnings.append(f"  • {var_name} - for {description}")
    
    if env_warnings:
        console.print("\n[yellow]⚠ Missing environment variables:[/yellow]")
        for warning in env_warnings:
            console.print(warning)
        console.print("\nSome tests may be skipped. Set these variables for complete recording.\n")
    
    # Run the command
    description = f"Recording cassettes for {service or 'all services'}"
    success = run_pytest_command(command, description)
    
    if success:
        cassettes_dir = get_cassettes_dir()
        if cassettes_dir.exists():
            cassette_count = len(list(cassettes_dir.glob("**/*.yaml")))
            console.print(f"\n[green]🎉 Recording complete! Generated {cassette_count} cassettes[/green]")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print("• Run unit tests: [cyan]ailego test unit[/cyan]")
        console.print("• Run all tests: [cyan]ailego test run[/cyan]")
        console.print("• Clean old cassettes: [cyan]ailego test clean-cassettes[/cyan]")
    else:
        console.print(f"\n[red]❌ Recording failed for {service or 'some services'}[/red]")
        raise typer.Exit(1)


def update_cassettes(
    service: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Update existing cassettes with new episodes.
    
    Args:
        service: Specific service to update
        verbose: Whether to show detailed output
    """
    console.print("\n[bold blue]🔄 Updating VCR Cassettes[/bold blue]")
    
    # Check if cassettes exist
    cassettes_dir = get_cassettes_dir()
    if not cassettes_dir.exists() or not list(cassettes_dir.glob("**/*.yaml")):
        console.print("[yellow]⚠ No existing cassettes found. Use 'record' command first.[/yellow]")
        raise typer.Exit(1)
    
    # Build pytest command  
    command = ["python", "-m", "pytest"]
    
    if service:
        integration_test = get_tests_dir() / "integration" / f"test_{service}_integration.py"
        if not integration_test.exists():
            console.print(f"[red]✗ Integration tests for '{service}' not found[/red]")
            raise typer.Exit(1)
        command.append(str(integration_test))
    else:
        command.append(str(get_tests_dir() / "integration"))
    
    # Use new_episodes mode to add new recordings while keeping existing ones
    command.extend(["--record-mode", "new_episodes"])
    
    if verbose:
        command.append("-v")
    
    command.extend(["-m", "integration"])
    
    description = f"Updating cassettes for {service or 'all services'}"
    success = run_pytest_command(command, description)
    
    if success:
        console.print(f"\n[green]🎉 Update complete![/green]")
    else:
        console.print(f"\n[red]❌ Update failed[/red]")
        raise typer.Exit(1)


def run_unit_tests(
    service: Optional[str] = None,
    verbose: bool = False,
    coverage: bool = False
) -> None:
    """
    Run unit tests using recorded cassettes.
    
    Args:
        service: Specific service to test
        verbose: Whether to show detailed output
        coverage: Whether to generate coverage report
    """
    console.print("\n[bold blue]🧪 Running Unit Tests[/bold blue]")
    
    # Check if cassettes exist
    cassettes_dir = get_cassettes_dir()
    if not cassettes_dir.exists() or not list(cassettes_dir.glob("**/*.yaml")):
        console.print("[yellow]⚠ No cassettes found. Run 'record' command first.[/yellow]")
        raise typer.Exit(1)
    
    # Build pytest command
    command = ["python", "-m", "pytest"]
    
    if service:
        unit_test = get_tests_dir() / "unit" / f"test_{service}_unit.py"
        if not unit_test.exists():
            console.print(f"[red]✗ Unit tests for '{service}' not found[/red]")
            raise typer.Exit(1)
        command.append(str(unit_test))
    else:
        command.append(str(get_tests_dir() / "unit"))
    
    # Always use none mode for unit tests (no recording)
    command.extend(["--record-mode", "none"])
    
    if verbose:
        command.append("-v")
    
    if coverage:
        command.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    command.extend(["-m", "unit"])
    
    description = f"Running unit tests for {service or 'all services'}"
    success = run_pytest_command(command, description)
    
    if success:
        console.print(f"\n[green]🎉 Unit tests passed![/green]")
        if coverage:
            console.print("Coverage report generated in [cyan]htmlcov/[/cyan]")
    else:
        console.print(f"\n[red]❌ Unit tests failed[/red]")
        raise typer.Exit(1)


def run_integration_tests(
    service: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Run integration tests (with real API calls).
    
    Args:
        service: Specific service to test
        verbose: Whether to show detailed output
    """
    console.print("\n[bold blue]🔗 Running Integration Tests[/bold blue]")
    
    # Build pytest command
    command = ["python", "-m", "pytest"]
    
    if service:
        integration_test = get_tests_dir() / "integration" / f"test_{service}_integration.py"
        if not integration_test.exists():
            console.print(f"[red]✗ Integration tests for '{service}' not found[/red]")
            raise typer.Exit(1)
        command.append(str(integration_test))
    else:
        command.append(str(get_tests_dir() / "integration"))
    
    # Use none mode to avoid recording during testing
    command.extend(["--record-mode", "none"])
    
    if verbose:
        command.append("-v")
    
    command.extend(["-m", "integration"])
    
    description = f"Running integration tests for {service or 'all services'}"
    success = run_pytest_command(command, description)
    
    if success:
        console.print(f"\n[green]🎉 Integration tests passed![/green]")
    else:
        console.print(f"\n[red]❌ Integration tests failed[/red]")
        raise typer.Exit(1)


def run_all_tests(verbose: bool = False, coverage: bool = False) -> None:
    """
    Run all tests (unit + integration).
    
    Args:
        verbose: Whether to show detailed output
        coverage: Whether to generate coverage report
    """
    console.print("\n[bold blue]🧪 Running All Tests[/bold blue]")
    
    # Build pytest command for all tests
    command = ["python", "-m", "pytest", str(get_tests_dir())]
    
    # Use none mode to avoid any recording
    command.extend(["--record-mode", "none"])
    
    if verbose:
        command.append("-v")
    
    if coverage:
        command.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    description = "Running all tests"
    success = run_pytest_command(command, description)
    
    if success:
        console.print(f"\n[green]🎉 All tests passed![/green]")
        if coverage:
            console.print("Coverage report generated in [cyan]htmlcov/[/cyan]")
    else:
        console.print(f"\n[red]❌ Some tests failed[/red]")
        raise typer.Exit(1)


def clean_cassettes(
    service: Optional[str] = None,
    confirm: bool = False
) -> None:
    """
    Clean old or invalid VCR cassettes.
    
    Args:
        service: Specific service to clean
        confirm: Skip confirmation prompt
    """
    console.print("\n[bold blue]🧹 Cleaning VCR Cassettes[/bold blue]")
    
    cassettes_dir = get_cassettes_dir()
    if not cassettes_dir.exists():
        console.print("[yellow]No cassettes directory found.[/yellow]")
        return
    
    # Find cassettes to clean
    if service:
        service_dir = cassettes_dir / service
        if service_dir.exists():
            cassettes = list(service_dir.glob("**/*.yaml"))
        else:
            console.print(f"[yellow]No cassettes found for service '{service}'[/yellow]")
            return
    else:
        cassettes = list(cassettes_dir.glob("**/*.yaml"))
    
    if not cassettes:
        console.print("[yellow]No cassettes found to clean.[/yellow]")
        return
    
    console.print(f"Found {len(cassettes)} cassettes to clean:")
    for cassette in cassettes[:10]:  # Show first 10
        console.print(f"  • {cassette.relative_to(cassettes_dir)}")
    
    if len(cassettes) > 10:
        console.print(f"  ... and {len(cassettes) - 10} more")
    
    if not confirm:
        if not typer.confirm(f"\nAre you sure you want to delete {len(cassettes)} cassettes?"):
            console.print("[yellow]Cleaning cancelled.[/yellow]")
            return
    
    # Delete cassettes
    deleted_count = 0
    for cassette in cassettes:
        try:
            cassette.unlink()
            deleted_count += 1
        except Exception as e:
            console.print(f"[red]Error deleting {cassette}: {e}[/red]")
    
    console.print(f"\n[green]🗑 Deleted {deleted_count} cassettes[/green]")
    
    # Clean empty directories
    for directory in cassettes_dir.rglob("*"):
        if directory.is_dir() and not list(directory.iterdir()):
            try:
                directory.rmdir()
            except:
                pass


def show_cassette_info() -> None:
    """Show information about existing cassettes."""
    console.print("\n[bold blue]📼 VCR Cassette Information[/bold blue]")
    
    cassettes_dir = get_cassettes_dir()
    if not cassettes_dir.exists():
        console.print("[yellow]No cassettes directory found.[/yellow]")
        return
    
    # Collect cassette statistics
    cassettes = list(cassettes_dir.glob("**/*.yaml"))
    if not cassettes:
        console.print("[yellow]No cassettes found.[/yellow]")
        return
    
    # Group by service
    services = {}
    for cassette in cassettes:
        # Extract service name from path
        parts = cassette.relative_to(cassettes_dir).parts
        service = parts[0] if parts else "unknown"
        
        if service not in services:
            services[service] = []
        services[service].append(cassette)
    
    # Create table
    table = Table(title="VCR Cassettes")
    table.add_column("Service", style="cyan")
    table.add_column("Cassettes", style="green")
    table.add_column("Total Size", style="yellow")
    table.add_column("Last Modified", style="magenta")
    
    total_cassettes = 0
    total_size = 0
    
    for service, cassette_list in sorted(services.items()):
        count = len(cassette_list)
        total_cassettes += count
        
        # Calculate total size
        service_size = sum(c.stat().st_size for c in cassette_list)
        total_size += service_size
        
        # Find most recent modification
        latest = max(c.stat().st_mtime for c in cassette_list)
        latest_str = Path(str(latest)).stem  # Basic formatting
        
        # Format size
        if service_size < 1024:
            size_str = f"{service_size} B"
        elif service_size < 1024 * 1024:
            size_str = f"{service_size / 1024:.1f} KB"
        else:
            size_str = f"{service_size / (1024 * 1024):.1f} MB"
        
        table.add_row(service, str(count), size_str, latest_str)
    
    console.print(table)
    
    # Summary
    if total_size < 1024:
        total_size_str = f"{total_size} B"
    elif total_size < 1024 * 1024:
        total_size_str = f"{total_size / 1024:.1f} KB"
    else:
        total_size_str = f"{total_size / (1024 * 1024):.1f} MB"
    
    console.print(f"\n[bold]Total: {total_cassettes} cassettes, {total_size_str}[/bold]")


def validate_cassettes() -> None:
    """Validate that cassettes don't contain sensitive data."""
    console.print("\n[bold blue]🔍 Validating VCR Cassettes[/bold blue]")
    
    cassettes_dir = get_cassettes_dir()
    if not cassettes_dir.exists():
        console.print("[yellow]No cassettes directory found.[/yellow]")
        return
    
    cassettes = list(cassettes_dir.glob("**/*.yaml"))
    if not cassettes:
        console.print("[yellow]No cassettes found.[/yellow]")
        return
    
    # Sensitive patterns to check for
    sensitive_patterns = [
        "sk-",  # OpenAI API keys
        "sk-ant-",  # Anthropic API keys
        "Bearer ",  # Bearer tokens
        "Basic ",  # Basic auth
        "password",  # Passwords
        "secret",  # Secrets
        "token",  # Tokens
    ]
    
    # IP address patterns - check for any non-localhost IP addresses
    import re
    # Match any IPv4 address except localhost (127.0.0.1) and 0.0.0.0
    ip_pattern = re.compile(r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)')
    localhost_pattern = re.compile(r'127\.0\.0\.1|0\.0\.0\.0')
    
    issues_found = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Validating {len(cassettes)} cassettes...", total=None)
        
        for cassette in cassettes:
            try:
                content = cassette.read_text(encoding='utf-8')
                
                # Check for sensitive patterns
                for pattern in sensitive_patterns:
                    if pattern.lower() in content.lower():
                        issues_found.append({
                            "cassette": cassette.relative_to(cassettes_dir),
                            "pattern": pattern,
                            "line": "unknown"  # Could be enhanced to find line numbers
                        })
                
                # Check for IP addresses (excluding localhost)
                ip_matches = ip_pattern.findall(content)
                for match in set(ip_matches):  # Remove duplicates
                    # Skip localhost addresses
                    if not localhost_pattern.match(match):
                        issues_found.append({
                            "cassette": cassette.relative_to(cassettes_dir),
                            "pattern": f"IP_ADDRESS: {match}",
                            "line": "unknown"
                        })
                        
            except Exception as e:
                issues_found.append({
                    "cassette": cassette.relative_to(cassettes_dir),
                    "pattern": "READ_ERROR",
                    "line": str(e)
                })
        
        progress.update(task, completed=True)
    
    if issues_found:
        console.print(f"\n[red]⚠ Found {len(issues_found)} potential security issues:[/red]")
        
        for issue in issues_found[:10]:  # Show first 10
            console.print(f"  • {issue['cassette']}: [yellow]{issue['pattern']}[/yellow]")
        
        if len(issues_found) > 10:
            console.print(f"  ... and {len(issues_found) - 10} more")
        
        console.print("\n[red]Please review these cassettes and re-record if needed.[/red]")
        console.print("Use [cyan]ailego test clean-cassettes[/cyan] to remove them.")
    else:
        console.print(f"\n[green]✓ All {len(cassettes)} cassettes look clean![/green]")