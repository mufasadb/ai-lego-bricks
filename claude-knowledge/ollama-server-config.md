# Ollama Server Configuration

## Remote Server Details
- **Server URL**: ollama.beachysapp.com
- **Port**: Default (11434)
- **Full URL**: http://ollama.beachysapp.com:11434
- **Local URL**: http://192.168.0.15:11434 (when on local network)

## Available Models
- **Vision-Language Model**: qwen2.5vlv (or similar variant)
- **Text Models**: mistral, llama variants, etc.
- **Recommended Models**: Use latest versions for best performance

## Configuration
When using Ollama in the codebase, set:
```bash
OLLAMA_URL=http://ollama.beachysapp.com:11434
```

For local development:
```bash
OLLAMA_URL=http://192.168.0.15:11434
```

## Usage Notes
- Server runs on Unraid with Cloudflare routing
- Both vision-language and text models are available
- Use for testing multi-model agents and workflows
- Server automatically available via custom URL routing
- Supports concurrent model usage for complex workflows

## Integration with AI Lego Bricks
- Used by OllamaTextClient in `llm/text_clients.py`
- Supports structured responses via LLM wrapper
- Compatible with Universal Tool Service
- Integrated with Agent Orchestration system