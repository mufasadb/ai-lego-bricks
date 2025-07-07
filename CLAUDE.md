# Claude Code Guidelines for AI Lego Bricks

## Project Overview
AI Lego Bricks is a modular library of building blocks for LLM agentic work, designed to be combined and configured like building blocks.

## Development Style
The goal here is to have Claude assist the user in learning how to utilise the tools in this project, so each task should be broken down and steppe through so that the user has an opportunity to learn whats going on and could feasibly continue development themselves without Claude afterward

## Partnership Approach
Claude Code should act as a thoughtful development partner, not just executing instructions blindly. Always:
- Critically evaluate each task for context and potential unintended impacts
- Ask clarifying questions when requirements are unclear or seem problematic
- Consider how changes fit within the broader application architecture
- Start every task by crafting a todo list to clarify intentions and next steps

## Implementation Standards
- Consider upcoming work during implementation but avoid leaving TODO comments in code
- Avoid mocking data unless explicitly instructed
- Write production-ready code, not placeholders
- Follow existing code patterns and conventions

## Completing Tasks
- Finish work during a task, if there is an error or compatibility issue, seek to fix it now rather than put it off or fake the passing of a test

## Context Gathering
- There is a claude-knlowleddge folder that is a folder built for claude to gather and store context about the repo
- For tool use, always check context7 for information on how to utilise an API or tool
- Check MCP/tools available and make use of them when need be

## MCP Integration
AI Lego Bricks now includes full MCP (Model Context Protocol) integration:
- **Server Management**: Start/stop MCP servers with process lifecycle management
- **Tool Discovery**: Automatic discovery and conversion of MCP tools to universal format
- **Credential Integration**: Secure credential management using CredentialManager
- **Multi-Provider**: MCP tools work with all LLM providers (OpenAI, Anthropic, Gemini, Ollama)
- **Agent Integration**: MCP tools automatically available in agent workflows

### Key Components
- `tools/mcp_types.py` - MCP protocol types and interfaces
- `tools/mcp_server_manager.py` - Process management and JSON-RPC communication
- `tools/mcp_tool_executor.py` - Integration with existing tool system
- `tools/mcp_config.py` - Configuration management with credential support

### Usage Pattern
```python
# Configure MCP server with credentials
config = MCPServerConfig(
    name="github",
    command=["npx", "-y", "@modelcontextprotocol/server-github"],
    env_credentials={"GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN"},
    required_credentials=["GITHUB_TOKEN"]
)

# Initialize with credential manager
await initialize_mcp_servers_from_config(credential_manager=cred_manager)
await register_mcp_tools_globally()
```

### Security Note
MCP servers use the same secure credential patterns as other tools - no hardcoded API keys in config files.

## Debugging
When debugging requires generating .py test, write them and them, use them, then delete them rather than leaving the root folder polluted with test .pys