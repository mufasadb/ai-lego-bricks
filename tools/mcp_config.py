"""
MCP server configuration management and auto-discovery.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from .mcp_types import MCPServerConfig, MCPTransport
from .mcp_server_manager import get_global_mcp_manager

logger = logging.getLogger(__name__)


class MCPConfigManager:
    """Manages MCP server configurations."""

    DEFAULT_CONFIG_PATHS = [
        "~/.config/mcp/servers.json",
        "~/.config/mcp/servers.yaml",
        "./mcp_servers.json",
        "./mcp_servers.yaml",
        "./.mcp/servers.json",
        "./.mcp/servers.yaml",
    ]

    def __init__(self, config_paths: Optional[List[str]] = None):
        """
        Initialize MCP config manager.

        Args:
            config_paths: Custom config file paths to check
        """
        self.config_paths = config_paths or self.DEFAULT_CONFIG_PATHS
        self._loaded_configs: Dict[str, MCPServerConfig] = {}

    async def load_configs(self) -> Dict[str, MCPServerConfig]:
        """Load MCP server configurations from config files."""
        configs = {}

        for config_path in self.config_paths:
            try:
                path = Path(config_path).expanduser()
                if path.exists():
                    logger.info(f"Loading MCP config from: {path}")
                    file_configs = await self._load_config_file(path)
                    configs.update(file_configs)
            except Exception as e:
                logger.error(f"Failed to load MCP config from {config_path}: {e}")

        # Also check for Claude Desktop config
        try:
            claude_configs = await self._load_claude_desktop_config()
            configs.update(claude_configs)
        except Exception as e:
            logger.debug(f"Could not load Claude Desktop config: {e}")

        self._loaded_configs = configs
        logger.info(f"Loaded {len(configs)} MCP server configurations")
        return configs

    async def _load_config_file(self, path: Path) -> Dict[str, MCPServerConfig]:
        """Load configuration from a single file."""
        configs = {}

        try:
            with open(path, "r") as f:
                if path.suffix.lower() in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Handle different config formats
            if isinstance(data, dict):
                if "servers" in data:
                    # Standard format: {"servers": {"name": {...}}}
                    server_data = data["servers"]
                else:
                    # Direct format: {"name": {...}}
                    server_data = data

                for name, config in server_data.items():
                    try:
                        mcp_config = self._parse_server_config(name, config)
                        configs[name] = mcp_config
                    except Exception as e:
                        logger.error(f"Failed to parse config for server '{name}': {e}")

        except Exception as e:
            logger.error(f"Failed to load config file {path}: {e}")

        return configs

    def _parse_server_config(
        self, name: str, config: Dict[str, Any]
    ) -> MCPServerConfig:
        """Parse a single server configuration."""
        # Handle different command formats
        command = config.get("command")
        if isinstance(command, str):
            command = [command]
        elif not isinstance(command, list):
            raise ValueError(f"Invalid command format for server '{name}': {command}")

        return MCPServerConfig(
            name=name,
            command=command,
            args=config.get("args"),
            env=config.get("env"),
            env_credentials=config.get("env_credentials"),
            transport=MCPTransport(config.get("transport", "stdio")),
            timeout=config.get("timeout", 30),
            auto_restart=config.get("auto_restart", True),
            working_directory=config.get("working_directory"),
            required_credentials=config.get("required_credentials"),
        )

    async def _load_claude_desktop_config(self) -> Dict[str, MCPServerConfig]:
        """Load MCP servers from Claude Desktop configuration."""
        configs = {}

        # Try common Claude Desktop config locations
        claude_config_paths = [
            "~/.config/claude/claude_desktop_config.json",
            "~/Library/Application Support/Claude/claude_desktop_config.json",
            os.path.expanduser("~")
            + "/AppData/Roaming/Claude/claude_desktop_config.json",
        ]

        for config_path in claude_config_paths:
            try:
                path = Path(config_path).expanduser()
                if path.exists():
                    logger.debug(f"Found Claude Desktop config at: {path}")

                    with open(path, "r") as f:
                        data = json.load(f)

                    mcp_servers = data.get("mcpServers", {})
                    for name, config in mcp_servers.items():
                        try:
                            # Convert Claude Desktop format to our format
                            mcp_config = self._parse_claude_desktop_server(name, config)
                            configs[f"claude_{name}"] = mcp_config
                        except Exception as e:
                            logger.error(
                                f"Failed to parse Claude Desktop server '{name}': {e}"
                            )

                    break  # Use first found config

            except Exception as e:
                logger.debug(
                    f"Could not read Claude Desktop config from {config_path}: {e}"
                )

        if configs:
            logger.info(f"Loaded {len(configs)} MCP servers from Claude Desktop config")

        return configs

    def _parse_claude_desktop_server(
        self, name: str, config: Dict[str, Any]
    ) -> MCPServerConfig:
        """Parse Claude Desktop MCP server configuration."""
        command = config.get("command")
        if isinstance(command, str):
            command = [command]

        args = config.get("args", [])
        env = config.get("env", {})

        return MCPServerConfig(
            name=name,
            command=command,
            args=args,
            env=env,
            transport=MCPTransport.STDIO,  # Claude Desktop uses stdio
            timeout=30,
            auto_restart=True,
        )

    async def save_config(self, path: str, configs: Dict[str, MCPServerConfig]) -> None:
        """Save MCP server configurations to a file."""
        config_path = Path(path).expanduser()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "servers": {
                name: {
                    "command": config.command,
                    "args": config.args,
                    "env": config.env,
                    "env_credentials": config.env_credentials,
                    "transport": config.transport.value,
                    "timeout": config.timeout,
                    "auto_restart": config.auto_restart,
                    "working_directory": config.working_directory,
                    "required_credentials": config.required_credentials,
                }
                for name, config in configs.items()
            }
        }

        with open(config_path, "w") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)

        logger.info(f"Saved MCP configuration to {config_path}")

    async def add_server_config(self, config: MCPServerConfig) -> None:
        """Add a new server configuration."""
        self._loaded_configs[config.name] = config

    async def remove_server_config(self, name: str) -> bool:
        """Remove a server configuration."""
        return self._loaded_configs.pop(name, None) is not None

    async def get_server_config(self, name: str) -> Optional[MCPServerConfig]:
        """Get a specific server configuration."""
        return self._loaded_configs.get(name)

    async def list_server_configs(self) -> List[str]:
        """List all configured server names."""
        return list(self._loaded_configs.keys())

    def get_loaded_configs(self) -> Dict[str, MCPServerConfig]:
        """Get all loaded configurations."""
        return self._loaded_configs.copy()


async def initialize_mcp_servers_from_config(
    config_manager: Optional[MCPConfigManager] = None,
    credential_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Initialize MCP servers from configuration files.

    Args:
        config_manager: Optional config manager instance
        credential_manager: Optional credential manager for secure credential handling

    Returns:
        Initialization results
    """
    if config_manager is None:
        config_manager = MCPConfigManager()

    # Load configurations
    configs = await config_manager.load_configs()

    if not configs:
        logger.info("No MCP server configurations found")
        return {"servers": [], "errors": []}

    # Get server manager with credential support
    from .mcp_server_manager import MCPServerManager

    if credential_manager:
        manager = MCPServerManager(credential_manager)
    else:
        manager = await get_global_mcp_manager()

    results = {"servers": [], "errors": [], "total": len(configs)}

    # Start each configured server
    for name, config in configs.items():
        try:
            await manager.add_server(config)
            results["servers"].append(
                {"name": name, "command": config.command, "status": "started"}
            )
            logger.info(f"Started MCP server '{name}'")
        except Exception as e:
            error_msg = f"Failed to start MCP server '{name}': {e}"
            logger.error(error_msg)
            results["errors"].append({"name": name, "error": error_msg})

    logger.info(
        f"MCP server initialization complete: {len(results['servers'])} started, {len(results['errors'])} failed"
    )
    return results


def create_example_config() -> Dict[str, Any]:
    """Create an example MCP server configuration."""
    return {
        "servers": {
            "filesystem": {
                "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
                "args": ["/path/to/allowed/directory"],
                "env": {},
                "transport": "stdio",
                "timeout": 30,
                "auto_restart": True,
            },
            "brave_search": {
                "command": ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
                "env": {"BRAVE_API_KEY": "your-api-key-here"},
            },
            "git": {
                "command": ["npx", "-y", "@modelcontextprotocol/server-git"],
                "args": ["--repository", "/path/to/git/repo"],
            },
        }
    }


# Global config manager instance
_global_config_manager = MCPConfigManager()


async def get_global_mcp_config_manager() -> MCPConfigManager:
    """Get the global MCP config manager."""
    return _global_config_manager
