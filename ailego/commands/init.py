"""
Project initialization command for AI Lego Bricks CLI.

This module handles creating new AI agent projects with templates and setup.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any
import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()


def init_project(project_name: str, template: str = "basic", force: bool = False):
    """
    Initialize a new AI Lego Bricks project.
    
    Args:
        project_name: Name of the project to create
        template: Template to use (basic, advanced, research)
        force: Whether to overwrite existing project
    """
    project_path = Path(project_name)
    
    # Check if project already exists
    if project_path.exists() and not force:
        if not Confirm.ask(f"Project '{project_name}' already exists. Overwrite?"):
            console.print("[yellow]Project initialization cancelled[/yellow]")
            return
    
    # Create project directory
    project_path.mkdir(exist_ok=True)
    
    console.print(f"[green]Initializing project: {project_name}[/green]")
    console.print(f"[blue]Using template: {template}[/blue]")
    
    # Copy template files
    _copy_template_files(project_path, template)
    
    # Create .env file
    _create_env_file(project_path)
    
    # Create project structure
    _create_project_structure(project_path)
    
    # Create sample agents
    _create_sample_agents(project_path, template)
    
    console.print(Panel(
        f"""[bold green]Project '{project_name}' initialized successfully![/bold green]

Next steps:
1. cd {project_name}
2. Edit .env with your API keys
3. pip install -e .
4. ailego verify
5. ailego run agents/simple_chat.json

[blue]Available commands:[/blue]
• ailego verify - Check your setup
• ailego run <agent.json> - Run an agent
• ailego create <type> - Create new agents
• ailego status - Check system status
""",
        title="Project Ready",
        border_style="green"
    ))


def _copy_template_files(project_path: Path, template: str):
    """Copy template files to the new project."""
    # Create basic structure
    (project_path / "agents").mkdir(exist_ok=True)
    (project_path / "data").mkdir(exist_ok=True)
    (project_path / "output").mkdir(exist_ok=True)
    (project_path / "prompts").mkdir(exist_ok=True)
    
    # Create README
    readme_content = _get_readme_template(project_path.name, template)
    (project_path / "README.md").write_text(readme_content)
    
    # Create CLAUDE.md for project instructions
    claude_md_content = _get_claude_md_template(project_path.name)
    (project_path / "CLAUDE.md").write_text(claude_md_content)


def _create_env_file(project_path: Path):
    """Create a .env file with common environment variables."""
    env_content = """# AI Lego Bricks Configuration

# LLM Providers
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Memory Services
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here

# Neo4j (Optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# TTS Services (Optional)
GOOGLE_APPLICATION_CREDENTIALS=path/to/google-credentials.json

# Ollama (Optional)
OLLAMA_URL=http://localhost:11434
"""
    
    (project_path / ".env").write_text(env_content)
    
    # Create .env.example
    (project_path / ".env.example").write_text(env_content)


def _create_project_structure(project_path: Path):
    """Create the basic project structure."""
    # Create pyproject.toml for the project
    pyproject_content = f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_path.name}"
version = "0.1.0"
description = "AI agent project built with AI Lego Bricks"
dependencies = [
    "ai-lego-bricks>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
]
"""
    
    (project_path / "pyproject.toml").write_text(pyproject_content)
    
    # Create .gitignore
    gitignore_content = """# AI Lego Bricks
.env
output/
data/private/
*.wav
*.mp3

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
"""
    
    (project_path / ".gitignore").write_text(gitignore_content)


def _create_sample_agents(project_path: Path, template: str):
    """Create sample agent configurations."""
    agents_dir = project_path / "agents"
    
    # Simple chat agent
    simple_chat = {
        "name": "simple_chat",
        "description": "A basic conversational agent",
        "config": {
            "default_llm_provider": "gemini"
        },
        "steps": [
            {
                "id": "get_input",
                "type": "input",
                "config": {"prompt": "What can I help you with?"},
                "outputs": ["user_query"]
            },
            {
                "id": "generate_response",
                "type": "llm_chat",
                "inputs": {
                    "message": {"from_step": "get_input", "field": "user_query"}
                },
                "config": {
                    "system_message": "You are a helpful AI assistant."
                },
                "outputs": ["response"]
            },
            {
                "id": "output",
                "type": "output",
                "inputs": {
                    "result": {"from_step": "generate_response", "field": "response"}
                }
            }
        ]
    }
    
    (agents_dir / "simple_chat.json").write_text(json.dumps(simple_chat, indent=2))
    
    if template == "advanced":
        # Memory-enabled agent
        memory_agent = {
            "name": "memory_chat",
            "description": "Chat agent with memory capabilities",
            "config": {
                "default_llm_provider": "gemini",
                "memory_provider": "supabase"
            },
            "steps": [
                {
                    "id": "get_input",
                    "type": "input",
                    "config": {"prompt": "Ask me anything:"},
                    "outputs": ["user_query"]
                },
                {
                    "id": "retrieve_memory",
                    "type": "memory_retrieve",
                    "inputs": {
                        "query": {"from_step": "get_input", "field": "user_query"}
                    },
                    "outputs": ["relevant_memories"]
                },
                {
                    "id": "generate_response",
                    "type": "llm_chat",
                    "inputs": {
                        "message": {"from_step": "get_input", "field": "user_query"},
                        "context": {"from_step": "retrieve_memory", "field": "relevant_memories"}
                    },
                    "config": {
                        "system_message": "You are a helpful assistant with access to previous conversations."
                    },
                    "outputs": ["response"]
                },
                {
                    "id": "store_memory",
                    "type": "memory_store",
                    "inputs": {
                        "content": {"from_step": "generate_response", "field": "response"},
                        "metadata": {"conversation_id": "user_session"}
                    }
                },
                {
                    "id": "output",
                    "type": "output",
                    "inputs": {
                        "result": {"from_step": "generate_response", "field": "response"}
                    }
                }
            ]
        }
        
        (agents_dir / "memory_chat.json").write_text(json.dumps(memory_agent, indent=2))


def _get_readme_template(project_name: str, template: str) -> str:
    """Get README template content."""
    return f"""# {project_name}

AI agent project built with AI Lego Bricks.

## Setup

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Verify setup:
   ```bash
   ailego verify
   ```

## Usage

Run your first agent:
```bash
ailego run agents/simple_chat.json
```

Create new agents:
```bash
ailego create chat --name "my-agent"
```

## Project Structure

- `agents/` - Agent configuration files
- `data/` - Input data and documents
- `output/` - Generated outputs and results
- `prompts/` - Reusable prompt templates

## Available Commands

- `ailego verify` - Check system setup
- `ailego run <agent.json>` - Execute an agent
- `ailego create <type>` - Create new agents
- `ailego status` - View system status

## Template: {template}

{_get_template_description(template)}
"""


def _get_template_description(template: str) -> str:
    """Get description for the chosen template."""
    descriptions = {
        "basic": "Simple chat and text processing agents with minimal dependencies.",
        "advanced": "Full-featured setup with memory, multi-modal, and advanced orchestration.",
        "research": "Specialized for document analysis, research, and knowledge extraction."
    }
    return descriptions.get(template, "Custom template configuration.")


def _get_claude_md_template(project_name: str) -> str:
    """Get CLAUDE.md template for project instructions."""
    return f"""# {project_name} - AI Agent Project

## Project Overview
This is an AI agent project built with AI Lego Bricks, focusing on modular and configurable AI workflows.

## Development Guidelines

### Agent Development
- Keep agents focused and modular
- Use JSON configuration for workflows
- Test agents incrementally
- Document agent purposes and inputs/outputs

### Code Organization
- `agents/` - Core agent configurations
- `data/` - Input data and test files
- `output/` - Generated results and artifacts
- `prompts/` - Reusable prompt templates

### Best Practices
- Use environment variables for sensitive data
- Version control agent configurations
- Test with minimal inputs first
- Document complex workflows

## Available Tools
- Memory services (Supabase, Neo4j)
- LLM providers (Gemini, Anthropic, Ollama)
- TTS services (OpenAI, Google)
- Document processing (PDF, text)
- Multi-modal capabilities (vision, audio)

## Quick Commands
- `ailego verify` - Check setup
- `ailego run <agent>` - Execute agent
- `ailego create <type>` - Generate new agent
- `ailego status` - System status
"""