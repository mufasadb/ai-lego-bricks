"""
Example secure tools that demonstrate proper API key and secret handling.
"""
import asyncio
from typing import Dict, Optional
from .secure_tool_executor import APIToolExecutor, DatabaseToolExecutor, WebhookToolExecutor
from .tool_types import ToolSchema, ToolParameter, ParameterType, ToolCall, ToolResult, Tool

try:
    from credentials import CredentialManager
except ImportError:
    from .secure_tool_executor import CredentialManager

class OpenAIToolExecutor(APIToolExecutor):
    """Example tool that uses OpenAI API with proper key management."""
    
    def __init__(self, credential_manager: Optional[CredentialManager] = None,
                 model: str = "gpt-3.5-turbo"):
        self.model = model
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key_name="OPENAI_API_KEY",
            credential_manager=credential_manager
        )
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute OpenAI chat completion."""
        try:
            prompt = tool_call.parameters.get("prompt", "")
            max_tokens = tool_call.parameters.get("max_tokens", 150)
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            
            response = await self.make_api_request("chat/completions", "POST", data)
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result={
                    "response": response["choices"][0]["message"]["content"],
                    "model": self.model,
                    "usage": response.get("usage", {})
                }
            )
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )
    
    @classmethod
    def get_schema(cls) -> ToolSchema:
        return ToolSchema(
            name="openai_chat",
            description="Generate text using OpenAI's GPT models",
            parameters=ToolParameter(
                type=ParameterType.OBJECT,
                properties={
                    "prompt": ToolParameter(
                        type=ParameterType.STRING,
                        description="The prompt to send to the AI"
                    ),
                    "max_tokens": ToolParameter(
                        type=ParameterType.INTEGER,
                        description="Maximum tokens to generate"
                    )
                },
                required=["prompt"]
            )
        )

class SlackWebhookExecutor(WebhookToolExecutor):
    """Example tool that sends Slack messages via webhook."""
    
    def __init__(self, credential_manager: Optional[CredentialManager] = None):
        super().__init__(
            webhook_url_key="SLACK_WEBHOOK_URL",
            credential_manager=credential_manager
        )
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Send message to Slack."""
        try:
            message = tool_call.parameters.get("message", "")
            channel = tool_call.parameters.get("channel", "#general")
            username = tool_call.parameters.get("username", "AI Agent")
            
            data = {
                "text": message,
                "channel": channel,
                "username": username
            }
            
            response = await self.send_webhook(data)
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result={"status": "sent", "channel": channel, "response": response}
            )
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )
    
    @classmethod
    def get_schema(cls) -> ToolSchema:
        return ToolSchema(
            name="slack_message",
            description="Send a message to Slack via webhook",
            parameters=ToolParameter(
                type=ParameterType.OBJECT,
                properties={
                    "message": ToolParameter(
                        type=ParameterType.STRING,
                        description="The message to send"
                    ),
                    "channel": ToolParameter(
                        type=ParameterType.STRING,
                        description="Slack channel (optional, defaults to #general)"
                    ),
                    "username": ToolParameter(
                        type=ParameterType.STRING,
                        description="Username to display (optional)"
                    )
                },
                required=["message"]
            )
        )

class SupabaseQueryExecutor(DatabaseToolExecutor):
    """Example tool that queries Supabase with proper credential handling."""
    
    def __init__(self, credential_manager: Optional[CredentialManager] = None):
        super().__init__(
            connection_credentials=["SUPABASE_URL", "SUPABASE_ANON_KEY"],
            credential_manager=credential_manager
        )
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute Supabase query."""
        try:
            table = tool_call.parameters.get("table", "")
            query_type = tool_call.parameters.get("query_type", "select")
            filters = tool_call.parameters.get("filters", {})
            
            # Build Supabase REST API URL
            base_url = self.require_credential("SUPABASE_URL")
            api_key = self.require_credential("SUPABASE_ANON_KEY")
            
            url = f"{base_url}/rest/v1/{table}"
            headers = {
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Add filters as query parameters
            if filters and query_type == "select":
                filter_params = []
                for key, value in filters.items():
                    filter_params.append(f"{key}=eq.{value}")
                if filter_params:
                    url += "?" + "&".join(filter_params)
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                if query_type == "select":
                    async with session.get(url, headers=headers) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            raise Exception(f"Supabase query failed: {response.status} - {error_text}")
                        data = await response.json()
                else:
                    raise ValueError(f"Unsupported query type: {query_type}")
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result={
                    "table": table,
                    "query_type": query_type,
                    "results": data,
                    "count": len(data) if isinstance(data, list) else 1
                }
            )
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )
    
    @classmethod
    def get_schema(cls) -> ToolSchema:
        return ToolSchema(
            name="supabase_query",
            description="Query data from Supabase database",
            parameters=ToolParameter(
                type=ParameterType.OBJECT,
                properties={
                    "table": ToolParameter(
                        type=ParameterType.STRING,
                        description="Table name to query"
                    ),
                    "query_type": ToolParameter(
                        type=ParameterType.STRING,
                        description="Type of query (select, insert, update, delete)",
                        enum=["select"]  # Only implement select for this example
                    ),
                    "filters": ToolParameter(
                        type=ParameterType.OBJECT,
                        description="Filters to apply to the query"
                    )
                },
                required=["table"]
            )
        )

class GitHubAPIExecutor(APIToolExecutor):
    """Example tool that interacts with GitHub API."""
    
    def __init__(self, credential_manager: Optional[CredentialManager] = None):
        super().__init__(
            base_url="https://api.github.com",
            api_key_name="GITHUB_TOKEN",
            credential_manager=credential_manager,
            additional_headers={"Accept": "application/vnd.github.v3+json"}
        )
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute GitHub API operation."""
        try:
            operation = tool_call.parameters.get("operation", "get_repo")
            repo = tool_call.parameters.get("repo", "")
            
            if operation == "get_repo":
                if not repo:
                    raise ValueError("repo parameter required for get_repo operation")
                
                response = await self.make_api_request(f"repos/{repo}")
                
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={
                        "repo": repo,
                        "name": response["name"],
                        "description": response["description"],
                        "stars": response["stargazers_count"],
                        "forks": response["forks_count"],
                        "language": response["language"],
                        "url": response["html_url"]
                    }
                )
            
            elif operation == "list_issues":
                if not repo:
                    raise ValueError("repo parameter required for list_issues operation")
                
                response = await self.make_api_request(f"repos/{repo}/issues")
                
                issues = [
                    {
                        "number": issue["number"],
                        "title": issue["title"],
                        "state": issue["state"],
                        "author": issue["user"]["login"],
                        "url": issue["html_url"]
                    }
                    for issue in response[:10]  # Limit to first 10
                ]
                
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={"repo": repo, "issues": issues, "count": len(issues)}
                )
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e)
            )
    
    @classmethod
    def get_schema(cls) -> ToolSchema:
        return ToolSchema(
            name="github_api",
            description="Interact with GitHub repositories and issues",
            parameters=ToolParameter(
                type=ParameterType.OBJECT,
                properties={
                    "operation": ToolParameter(
                        type=ParameterType.STRING,
                        description="Operation to perform",
                        enum=["get_repo", "list_issues"]
                    ),
                    "repo": ToolParameter(
                        type=ParameterType.STRING,
                        description="Repository in format 'owner/repo'"
                    )
                },
                required=["operation", "repo"]
            )
        )

def create_secure_example_tools(credential_manager: Optional[CredentialManager] = None) -> Dict[str, Tool]:
    """Create example secure tools with credential management."""
    
    tools = {}
    
    # OpenAI tool (if API key available)
    try:
        openai_tool = Tool(
            schema=OpenAIToolExecutor.get_schema(),
            executor=OpenAIToolExecutor(credential_manager)
        )
        tools["openai_chat"] = openai_tool
    except ValueError as e:
        print(f"OpenAI tool not available: {e}")
    
    # Slack webhook tool (if webhook URL available)
    try:
        slack_tool = Tool(
            schema=SlackWebhookExecutor.get_schema(),
            executor=SlackWebhookExecutor(credential_manager)
        )
        tools["slack_message"] = slack_tool
    except ValueError as e:
        print(f"Slack tool not available: {e}")
    
    # Supabase tool (if credentials available)
    try:
        supabase_tool = Tool(
            schema=SupabaseQueryExecutor.get_schema(),
            executor=SupabaseQueryExecutor(credential_manager)
        )
        tools["supabase_query"] = supabase_tool
    except ValueError as e:
        print(f"Supabase tool not available: {e}")
    
    # GitHub tool (if token available)
    try:
        github_tool = Tool(
            schema=GitHubAPIExecutor.get_schema(),
            executor=GitHubAPIExecutor(credential_manager)
        )
        tools["github_api"] = github_tool
    except ValueError as e:
        print(f"GitHub tool not available: {e}")
    
    return tools

async def register_secure_tools_conditionally(category: str = "secure"):
    """Register secure tools only if their credentials are available."""
    from .tool_registry import register_tool_globally
    
    # Create credential manager that loads from environment
    cred_manager = CredentialManager(load_env=True)
    
    # Get available tools
    tools = create_secure_example_tools(cred_manager)
    
    print(f"Registering {len(tools)} secure tools with available credentials:")
    
    for name, tool in tools.items():
        await register_tool_globally(tool, category)
        print(f"  ✅ {tool.schema.name}: {tool.schema.description}")
    
    if not tools:
        print("  ❌ No secure tools available (missing API keys/credentials)")
    
    return tools

if __name__ == "__main__":
    asyncio.run(register_secure_tools_conditionally())