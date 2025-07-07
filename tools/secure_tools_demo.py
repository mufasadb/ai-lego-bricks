"""
Demonstration of secure tool usage with API keys and credential management.
"""
import asyncio
import json
from typing import Dict, Any, Optional

try:
    from credentials import CredentialManager
except ImportError:
    print("Warning: credentials module not found, using mock")
    class CredentialManager:
        def __init__(self, credentials=None, load_env=True):
            self.credentials = credentials or {}
        def get_credential(self, key, default=None):
            return self.credentials.get(key, default)

from .tool_service import ToolService
from .secure_example_tools import create_secure_example_tools

async def demo_credential_management():
    """Demonstrate different credential management patterns."""
    
    print("=== Credential Management Demo ===\n")
    
    # Pattern 1: Environment-based credentials (production pattern)
    print("1. Environment-based credentials (checks .env file)")
    env_creds = CredentialManager(load_env=True)
    print(f"   OpenAI API Key available: {bool(env_creds.get_credential('OPENAI_API_KEY'))}")
    print(f"   GitHub Token available: {bool(env_creds.get_credential('GITHUB_TOKEN'))}")
    print(f"   Slack Webhook available: {bool(env_creds.get_credential('SLACK_WEBHOOK_URL'))}")
    
    # Pattern 2: Explicit credentials (library/multi-tenant pattern)
    print("\n2. Explicit credentials (library-safe)")
    explicit_creds = CredentialManager({
        "OPENAI_API_KEY": "sk-fake-key-for-demo",
        "GITHUB_TOKEN": "ghp_fake_token_for_demo",
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/fake/webhook"
    }, load_env=False)
    
    # Pattern 3: Mixed credentials (override specific keys)
    print("\n3. Mixed credentials (explicit + environment)")
    mixed_creds = CredentialManager({
        "OPENAI_API_KEY": "sk-override-key"  # Override this specific key
        # Other keys come from environment
    }, load_env=True)
    
    return env_creds, explicit_creds, mixed_creds

async def demo_secure_tool_registration():
    """Demonstrate secure tool registration with credential validation."""
    
    print("\n=== Secure Tool Registration Demo ===\n")
    
    # Create credential manager with some fake credentials
    creds = CredentialManager({
        "OPENAI_API_KEY": "sk-fake-key",
        "GITHUB_TOKEN": "ghp_fake_token"
        # Note: Missing SLACK_WEBHOOK_URL and Supabase credentials
    }, load_env=False)
    
    # Create tool service with credential manager
    tool_service = ToolService(credential_manager=creds)
    
    # Try to create secure tools (some will fail due to missing credentials)
    tools = create_secure_example_tools(creds)
    
    print(f"Available tools with current credentials: {len(tools)}")
    for name, tool in tools.items():
        print(f"  ✅ {tool.schema.name}: {tool.schema.description}")
    
    # Register tools that are available
    for name, tool in tools.items():
        await tool_service.register_tool(tool, "secure")
    
    # Validate tool credentials
    validation_results = await tool_service.validate_tool_credentials()
    print(f"\nCredential validation results:")
    print(f"  Available tools: {validation_results['available_tools']}")
    print(f"  Unavailable tools: {validation_results['unavailable_tools']}")
    
    if validation_results['missing_credentials']:
        print(f"  Missing credentials: {validation_results['missing_credentials']}")
    
    return tool_service, validation_results

async def demo_workflow_with_secure_tools():
    """Demonstrate workflow configuration with secure tools."""
    
    print("\n=== Workflow with Secure Tools Demo ===\n")
    
    # Example workflow that uses secure tools
    workflow_config = {
        "name": "Secure Tools Workflow",
        "description": "Demonstrates secure tool usage in workflows",
        "credential_config": {
            "load_env": True,  # Load from .env
            "required_credentials": [
                "OPENAI_API_KEY",  # For OpenAI tool
                "GITHUB_TOKEN"     # For GitHub tool
            ]
        },
        "steps": [
            {
                "id": "github_lookup",
                "type": "tool_call",
                "description": "Look up repository information",
                "config": {
                    "provider": "ollama",
                    "model": "llama3.1:8b",
                    "tools": ["github_api"],
                    "tool_choice": "github_api",  # Force use of this tool
                    "auto_execute": True,
                    "prompt": "You are a GitHub repository assistant. Use the github_api tool to get information about repositories."
                },
                "inputs": {
                    "message": "Get information about the repository 'anthropics/claude-code'"
                },
                "outputs": ["repo_info"]
            },
            {
                "id": "ai_summary",
                "type": "tool_call", 
                "description": "Generate AI summary of the repository",
                "config": {
                    "provider": "openai",  # Use OpenAI for summary
                    "tools": ["openai_chat"],
                    "tool_choice": "openai_chat",
                    "auto_execute": True,
                    "prompt": "You are a repository analyst. Use the openai_chat tool to generate insights about repositories."
                },
                "inputs": {
                    "message": {
                        "from_step": "github_lookup",
                        "key": "repo_info",
                        "transform": "Create a summary and analysis of this GitHub repository: {repo_info}"
                    }
                },
                "outputs": ["ai_analysis"]
            },
            {
                "id": "notification",
                "type": "tool_call",
                "description": "Send notification about analysis",
                "config": {
                    "provider": "test",
                    "tools": ["slack_message"],
                    "tool_choice": "slack_message",
                    "auto_execute": True,
                    "prompt": "You are a notification assistant. Send appropriate Slack messages."
                },
                "inputs": {
                    "message": {
                        "from_step": "ai_summary", 
                        "key": "ai_analysis",
                        "transform": "Send a Slack notification about this repository analysis: {ai_analysis}"
                    }
                },
                "outputs": ["notification_status"]
            }
        ]
    }
    
    print("Example workflow configuration:")
    print(json.dumps(workflow_config, indent=2))
    
    return workflow_config

async def demo_credential_isolation():
    """Demonstrate credential isolation for multi-tenant scenarios."""
    
    print("\n=== Credential Isolation Demo ===\n")
    
    # Simulate different tenants with different API keys
    tenant_a_creds = CredentialManager({
        "OPENAI_API_KEY": "sk-tenant-a-key",
        "GITHUB_TOKEN": "ghp_tenant_a_token"
    }, load_env=False)
    
    tenant_b_creds = CredentialManager({
        "OPENAI_API_KEY": "sk-tenant-b-key", 
        "GITHUB_TOKEN": "ghp_tenant_b_token"
    }, load_env=False)
    
    # Create isolated tool services
    tenant_a_service = ToolService(credential_manager=tenant_a_creds)
    tenant_b_service = ToolService(credential_manager=tenant_b_creds)
    
    print("Tenant A tools:")
    tenant_a_tools = create_secure_example_tools(tenant_a_creds)
    for name, tool in tenant_a_tools.items():
        await tenant_a_service.register_tool(tool, f"tenant_a")
        print(f"  ✅ {tool.schema.name} (Tenant A credentials)")
    
    print("\nTenant B tools:")
    tenant_b_tools = create_secure_example_tools(tenant_b_creds)
    for name, tool in tenant_b_tools.items():
        await tenant_b_service.register_tool(tool, f"tenant_b")
        print(f"  ✅ {tool.schema.name} (Tenant B credentials)")
    
    # Show that credentials are isolated
    print(f"\nTenant A OpenAI key: {tenant_a_creds.get_credential('OPENAI_API_KEY')}")
    print(f"Tenant B OpenAI key: {tenant_b_creds.get_credential('OPENAI_API_KEY')}")
    
    return tenant_a_service, tenant_b_service

async def demo_credential_validation():
    """Demonstrate credential validation and error handling."""
    
    print("\n=== Credential Validation Demo ===\n")
    
    # Test with missing credentials
    incomplete_creds = CredentialManager({
        "OPENAI_API_KEY": "sk-present-key"
        # Missing GITHUB_TOKEN, SLACK_WEBHOOK_URL, etc.
    }, load_env=False)
    
    tool_service = ToolService(credential_manager=incomplete_creds)
    
    # Try to register tools with credential validation
    tool_configs = [
        {
            "tool": create_secure_example_tools(incomplete_creds).get("openai_chat"),
            "category": "ai",
            "required_credentials": ["OPENAI_API_KEY"]
        }
    ]
    
    # This should work (OpenAI key is present)
    if tool_configs[0]["tool"]:
        results = await tool_service.register_tools_with_credentials(tool_configs)
        print("Registration results with complete credentials:")
        print(f"  Registered: {len(results['registered'])}")
        print(f"  Failed: {len(results['failed'])}")
    
    # Try with missing credentials
    try:
        incomplete_tools = create_secure_example_tools(CredentialManager({}, load_env=False))
        print("\nTools available with no credentials:", len(incomplete_tools))
    except Exception as e:
        print(f"\nExpected error with missing credentials: {e}")
    
    # Validation report
    validation = await tool_service.validate_tool_credentials()
    print(f"\nValidation summary:")
    print(f"  Status: {validation['status']}")
    print(f"  Available tools: {validation['available_tools']}")
    print(f"  Unavailable tools: {validation['unavailable_tools']}")

async def main():
    """Main demo function."""
    
    print("🔐 Secure Tool Service Demo")
    print("=" * 50)
    
    # Demo 1: Credential management patterns
    await demo_credential_management()
    
    # Demo 2: Secure tool registration
    await demo_secure_tool_registration()
    
    # Demo 3: Workflow configuration
    await demo_workflow_with_secure_tools()
    
    # Demo 4: Credential isolation
    await demo_credential_isolation()
    
    # Demo 5: Credential validation
    await demo_credential_validation()
    
    print("\n" + "=" * 50)
    print("✅ Secure Tool Service Demo Complete!")
    print("\nKey takeaways:")
    print("1. 🔑 Credentials are managed centrally with CredentialManager")
    print("2. 🛡️  Tools validate required credentials at registration")
    print("3. 🏢 Multi-tenant isolation is supported")
    print("4. ⚠️  Missing credentials are detected early with clear errors")
    print("5. 🔄 Different credential sources are supported (env, explicit, mixed)")
    print("6. 📋 Validation reports help with debugging credential issues")

if __name__ == "__main__":
    asyncio.run(main())