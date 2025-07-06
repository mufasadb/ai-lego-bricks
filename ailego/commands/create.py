"""
Agent creation command for AI Lego Bricks CLI.

This module handles creating new AI agent configurations with templates.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

console = Console()

# Agent templates
AGENT_TEMPLATES = {
    "chat": {
        "name": "Simple Chat Agent",
        "description": "Basic conversational agent with customizable personality",
        "template": {
            "name": "chat_agent",
            "description": "A conversational AI agent",
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
    },
    "document-analysis": {
        "name": "Document Analysis Agent",
        "description": "Analyzes documents and answers questions about them",
        "template": {
            "name": "document_analysis_agent",
            "description": "Analyzes documents and provides insights",
            "config": {
                "default_llm_provider": "gemini"
            },
            "steps": [
                {
                    "id": "load_document",
                    "type": "input",
                    "config": {"prompt": "Enter document path:"},
                    "outputs": ["document_path"]
                },
                {
                    "id": "process_document",
                    "type": "document_processing",
                    "inputs": {
                        "file_path": {"from_step": "load_document", "field": "document_path"}
                    },
                    "outputs": ["document_content"]
                },
                {
                    "id": "chunk_content",
                    "type": "chunk_text",
                    "inputs": {
                        "text": {"from_step": "process_document", "field": "document_content"}
                    },
                    "outputs": ["chunks"]
                },
                {
                    "id": "store_chunks",
                    "type": "memory_store",
                    "inputs": {
                        "content": {"from_step": "chunk_content", "field": "chunks"},
                        "metadata": {"document_type": "analysis"}
                    }
                },
                {
                    "id": "get_question",
                    "type": "input",
                    "config": {"prompt": "What would you like to know about the document?"},
                    "outputs": ["question"]
                },
                {
                    "id": "retrieve_relevant",
                    "type": "memory_retrieve",
                    "inputs": {
                        "query": {"from_step": "get_question", "field": "question"}
                    },
                    "outputs": ["relevant_chunks"]
                },
                {
                    "id": "analyze",
                    "type": "llm_chat",
                    "inputs": {
                        "message": {"from_step": "get_question", "field": "question"},
                        "context": {"from_step": "retrieve_relevant", "field": "relevant_chunks"}
                    },
                    "config": {
                        "system_message": "You are a document analysis expert. Use the provided context to answer questions accurately."
                    },
                    "outputs": ["analysis"]
                },
                {
                    "id": "output",
                    "type": "output",
                    "inputs": {
                        "result": {"from_step": "analyze", "field": "analysis"}
                    }
                }
            ]
        }
    },
    "research": {
        "name": "Research Agent",
        "description": "Conducts research across multiple sources and synthesizes findings",
        "template": {
            "name": "research_agent",
            "description": "Multi-source research and synthesis agent",
            "config": {
                "default_llm_provider": "gemini"
            },
            "steps": [
                {
                    "id": "get_research_topic",
                    "type": "input",
                    "config": {"prompt": "What topic would you like to research?"},
                    "outputs": ["research_topic"]
                },
                {
                    "id": "retrieve_sources",
                    "type": "memory_retrieve",
                    "inputs": {
                        "query": {"from_step": "get_research_topic", "field": "research_topic"}
                    },
                    "config": {"limit": 10},
                    "outputs": ["source_materials"]
                },
                {
                    "id": "analyze_sources",
                    "type": "llm_chat",
                    "inputs": {
                        "message": {"from_step": "get_research_topic", "field": "research_topic"},
                        "context": {"from_step": "retrieve_sources", "field": "source_materials"}
                    },
                    "config": {
                        "system_message": "You are a research analyst. Synthesize the provided sources to create a comprehensive research report."
                    },
                    "outputs": ["research_report"]
                },
                {
                    "id": "generate_structured_output",
                    "type": "llm_structured",
                    "inputs": {
                        "message": {"from_step": "analyze_sources", "field": "research_report"}
                    },
                    "config": {
                        "response_schema": {
                            "name": "ResearchReport",
                            "fields": {
                                "summary": {"type": "string"},
                                "key_findings": {"type": "list"},
                                "recommendations": {"type": "list"},
                                "sources_used": {"type": "list"},
                                "confidence_score": {"type": "float"}
                            }
                        }
                    },
                    "outputs": ["structured_report"]
                },
                {
                    "id": "output",
                    "type": "output",
                    "inputs": {
                        "result": {"from_step": "generate_structured_output", "field": "structured_report"}
                    }
                }
            ]
        }
    },
    "vision": {
        "name": "Vision Analysis Agent",
        "description": "Analyzes images and provides detailed descriptions",
        "template": {
            "name": "vision_agent",
            "description": "Image analysis and description agent",
            "config": {
                "default_llm_provider": "gemini"
            },
            "steps": [
                {
                    "id": "get_image",
                    "type": "input",
                    "config": {"prompt": "Enter image path:"},
                    "outputs": ["image_path"]
                },
                {
                    "id": "analyze_image",
                    "type": "llm_vision",
                    "inputs": {
                        "image_path": {"from_step": "get_image", "field": "image_path"},
                        "message": "Analyze this image in detail"
                    },
                    "config": {
                        "provider": "gemini"
                    },
                    "outputs": ["image_analysis"]
                },
                {
                    "id": "get_question",
                    "type": "input",
                    "config": {"prompt": "What specific question do you have about the image?"},
                    "outputs": ["question"]
                },
                {
                    "id": "answer_question",
                    "type": "llm_chat",
                    "inputs": {
                        "message": {"from_step": "get_question", "field": "question"},
                        "context": {"from_step": "analyze_image", "field": "image_analysis"}
                    },
                    "config": {
                        "system_message": "You are a vision analysis expert. Use the image analysis to answer questions accurately."
                    },
                    "outputs": ["answer"]
                },
                {
                    "id": "output",
                    "type": "output",
                    "inputs": {
                        "result": {"from_step": "answer_question", "field": "answer"}
                    }
                }
            ]
        }
    },
    "streaming": {
        "name": "Streaming Chat Agent",
        "description": "Real-time streaming conversation with optional TTS",
        "template": {
            "name": "streaming_agent",
            "description": "Streaming conversation agent with TTS support",
            "config": {
                "default_llm_provider": "ollama"
            },
            "steps": [
                {
                    "id": "get_input",
                    "type": "input",
                    "config": {"prompt": "Start a conversation:"},
                    "outputs": ["user_message"]
                },
                {
                    "id": "stream_response",
                    "type": "llm_chat",
                    "inputs": {
                        "message": {"from_step": "get_input", "field": "user_message"}
                    },
                    "config": {
                        "provider": "ollama",
                        "stream": True,
                        "system_message": "You are a helpful AI assistant. Respond naturally and conversationally."
                    },
                    "outputs": ["response"]
                },
                {
                    "id": "convert_to_speech",
                    "type": "tts",
                    "inputs": {
                        "text": {"from_step": "stream_response", "field": "response"}
                    },
                    "config": {
                        "provider": "auto",
                        "voice": "default",
                        "output_path": "output/response.wav"
                    },
                    "outputs": ["audio_file"]
                },
                {
                    "id": "output",
                    "type": "output",
                    "inputs": {
                        "result": {"from_step": "stream_response", "field": "response"},
                        "audio": {"from_step": "convert_to_speech", "field": "audio_file"}
                    }
                }
            ]
        }
    }
}


def create_agent(agent_type: str, name: Optional[str] = None, interactive: bool = True):
    """
    Create a new AI agent configuration.
    
    Args:
        agent_type: Type of agent to create
        name: Optional name for the agent
        interactive: Whether to use interactive mode
    """
    
    # Show available templates if invalid type
    if agent_type not in AGENT_TEMPLATES:
        console.print(f"[red]Unknown agent type: {agent_type}[/red]")
        _show_available_templates()
        return
    
    template_info = AGENT_TEMPLATES[agent_type]
    
    console.print(f"\n[bold blue]Creating {template_info['name']}[/bold blue]")
    console.print(f"[blue]{template_info['description']}[/blue]")
    
    # Get agent name
    if not name:
        name = Prompt.ask("Agent name", default=f"my_{agent_type}_agent")
    
    # Create base configuration
    agent_config = template_info["template"].copy()
    agent_config["name"] = name
    
    # Interactive customization
    if interactive:
        agent_config = _customize_agent_interactive(agent_config, agent_type)
    
    # Save agent configuration
    _save_agent_config(agent_config, name)


def _customize_agent_interactive(config: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
    """Interactively customize agent configuration."""
    
    console.print("\n[bold yellow]Customization Options:[/bold yellow]")
    
    # LLM Provider
    current_provider = config.get("config", {}).get("default_llm_provider", "gemini")
    new_provider = Prompt.ask(
        "LLM Provider",
        choices=["gemini", "anthropic", "ollama", "auto"],
        default=current_provider
    )
    
    if "config" not in config:
        config["config"] = {}
    config["config"]["default_llm_provider"] = new_provider
    
    # System message customization
    for step in config.get("steps", []):
        if step.get("type") == "llm_chat" and "system_message" in step.get("config", {}):
            current_system = step["config"]["system_message"]
            console.print(f"\n[blue]Current system message:[/blue] {current_system}")
            
            if Confirm.ask("Customize system message?"):
                new_system = Prompt.ask("New system message", default=current_system)
                step["config"]["system_message"] = new_system
    
    # Agent-specific customizations
    if agent_type == "streaming":
        # TTS configuration
        if Confirm.ask("Enable Text-to-Speech?"):
            _configure_tts_step(config)
    
    elif agent_type == "document-analysis":
        # Memory provider
        memory_provider = Prompt.ask(
            "Memory provider",
            choices=["supabase", "neo4j", "auto"],
            default="auto"
        )
        config["config"]["memory_provider"] = memory_provider
    
    return config


def _configure_tts_step(config: Dict[str, Any]):
    """Configure TTS step for streaming agents."""
    
    tts_provider = Prompt.ask(
        "TTS Provider",
        choices=["openai", "google", "coqui", "auto"],
        default="auto"
    )
    
    voice = Prompt.ask("Voice", default="default")
    output_path = Prompt.ask("Output path", default="output/response.wav")
    
    # Find and update TTS step
    for step in config.get("steps", []):
        if step.get("type") == "tts":
            step["config"]["provider"] = tts_provider
            step["config"]["voice"] = voice
            step["config"]["output_path"] = output_path
            break


def _save_agent_config(config: Dict[str, Any], name: str):
    """Save agent configuration to file."""
    
    # Ensure agents directory exists
    agents_dir = Path("agents")
    agents_dir.mkdir(exist_ok=True)
    
    # Generate filename
    filename = f"{name}.json"
    filepath = agents_dir / filename
    
    # Check if file exists
    if filepath.exists():
        if not Confirm.ask(f"Agent '{name}' already exists. Overwrite?"):
            console.print("[yellow]Agent creation cancelled[/yellow]")
            return
    
    # Save configuration
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        console.print(f"\n[green]✓ Agent created successfully![/green]")
        console.print(f"[blue]Location: {filepath}[/blue]")
        console.print(f"[blue]Test with: ailego run {filepath}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error saving agent: {e}[/red]")


def _show_available_templates():
    """Show available agent templates."""
    
    console.print("\n[bold blue]Available Agent Templates:[/bold blue]")
    
    table = Table()
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    
    for agent_type, template_info in AGENT_TEMPLATES.items():
        table.add_row(
            agent_type,
            template_info["name"],
            template_info["description"]
        )
    
    console.print(table)
    console.print("\n[blue]Usage: ailego create <type> --name <name>[/blue]")


def list_agent_types():
    """List available agent types."""
    return list(AGENT_TEMPLATES.keys())