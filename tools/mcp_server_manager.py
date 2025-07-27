"""
MCP server process management and communication.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .mcp_types import (
    MCPServerConfig,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPMethods,
    MCPInitializeParams,
    MCPClientCapabilities,
)

try:
    from credentials import CredentialManager
except ImportError:
    CredentialManager = None

logger = logging.getLogger(__name__)


class MCPServerProcess:
    """Manages a single MCP server process and communication."""

    def __init__(
        self,
        config: MCPServerConfig,
        credential_manager: Optional[CredentialManager] = None,
    ):
        self.config = config
        self.credential_manager = credential_manager
        self.process: Optional[asyncio.subprocess.Process] = None
        self.initialized = False
        self.capabilities = None
        self.server_info = None
        self._request_id_counter = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Validate credentials if required
        self._validate_credentials()

    def _validate_credentials(self):
        """Validate that all required credentials are available."""
        if not self.config.required_credentials:
            return

        if not self.credential_manager:
            if CredentialManager:
                self.credential_manager = CredentialManager()
            else:
                raise ValueError(
                    f"Server '{self.config.name}' requires credentials but no credential manager available"
                )

        missing = []
        for cred_key in self.config.required_credentials:
            if not self.credential_manager.get_credential(cred_key):
                missing.append(cred_key)

        if missing:
            raise ValueError(
                f"Missing required credentials for MCP server '{self.config.name}': {', '.join(missing)}"
            )

    def _prepare_environment(self) -> Optional[Dict[str, str]]:
        """Prepare environment variables for the server process."""
        import os

        env = os.environ.copy()

        # Add static environment variables
        if self.config.env:
            env.update(self.config.env)

        # Add credential-based environment variables
        if self.config.env_credentials and self.credential_manager:
            for env_var, cred_key in self.config.env_credentials.items():
                cred_value = self.credential_manager.get_credential(cred_key)
                if cred_value:
                    env[env_var] = cred_value
                else:
                    logger.warning(
                        f"Credential '{cred_key}' not found for env var '{env_var}'"
                    )

        return env

    async def start(self) -> None:
        """Start the MCP server process."""
        if self.process and self.process.returncode is None:
            return  # Already running

        try:
            # Build command
            command = self.config.command.copy()
            if self.config.args:
                command.extend(self.config.args)

            # Prepare environment with credentials
            env = self._prepare_environment()

            # Start process
            working_dir = self.config.working_directory
            if working_dir:
                working_dir = Path(working_dir).expanduser().resolve()

            logger.info(
                f"Starting MCP server '{self.config.name}' with command: {command}"
            )

            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=working_dir,
            )

            # Start reading responses
            self._read_task = asyncio.create_task(self._read_responses())

            # Initialize the server
            await self._initialize_server()

            logger.info(f"MCP server '{self.config.name}' started successfully")

        except Exception as e:
            logger.error(f"Failed to start MCP server '{self.config.name}': {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Force killing MCP server '{self.config.name}'")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping MCP server '{self.config.name}': {e}")
            finally:
                self.process = None

        # Cancel any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        self.initialized = False
        logger.info(f"MCP server '{self.config.name}' stopped")

    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Send a JSON-RPC request to the server."""
        if not self.process or self.process.returncode is not None:
            raise RuntimeError(f"MCP server '{self.config.name}' is not running")

        async with self._lock:
            self._request_id_counter += 1
            request_id = str(self._request_id_counter)

        request = JSONRPCRequest(id=request_id, method=method, params=params)

        # Create future for response
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future

        try:
            # Send request
            request_json = request.model_dump_json() + "\n"
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()

            # Wait for response
            timeout = timeout or self.config.timeout
            if timeout:
                response = await asyncio.wait_for(response_future, timeout=timeout)
            else:
                response = await response_future

            return response

        except asyncio.TimeoutError:
            logger.error(f"Request {method} to '{self.config.name}' timed out")
            raise
        except Exception as e:
            logger.error(
                f"Failed to send request {method} to '{self.config.name}': {e}"
            )
            raise
        finally:
            self._pending_requests.pop(request_id, None)

    async def _read_responses(self) -> None:
        """Read and process responses from the server."""
        try:
            while self.process and self.process.returncode is None:
                line = await self.process.stdout.readline()
                if not line:
                    break

                try:
                    response_data = json.loads(line.decode().strip())
                    await self._handle_response(response_data)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Invalid JSON from MCP server '{self.config.name}': {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing response from '{self.config.name}': {e}"
                    )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error reading from MCP server '{self.config.name}': {e}")

    async def _handle_response(self, response_data: Dict[str, Any]) -> None:
        """Handle a response from the server."""
        try:
            response = JSONRPCResponse(**response_data)

            # Handle response
            if response.id and response.id in self._pending_requests:
                future = self._pending_requests[response.id]
                if not future.done():
                    if response.error:
                        error = Exception(
                            f"MCP Error {response.error['code']}: {response.error['message']}"
                        )
                        future.set_exception(error)
                    else:
                        future.set_result(response.result)
            else:
                # Handle notification or unexpected response
                logger.debug(
                    f"Received notification from '{self.config.name}': {response_data}"
                )

        except Exception as e:
            logger.error(f"Error handling response from '{self.config.name}': {e}")

    async def _initialize_server(self) -> None:
        """Initialize the MCP server."""
        client_capabilities = MCPClientCapabilities(experimental={}, sampling={})

        params = MCPInitializeParams(
            capabilities=client_capabilities,
            clientInfo={"name": "AI-Lego-Bricks", "version": "1.0.0"},
        )

        try:
            result = await self.send_request(MCPMethods.INITIALIZE, params.model_dump())

            # Store server capabilities and info
            self.capabilities = result.get("capabilities", {})
            self.server_info = result.get("serverInfo", {})

            # Send initialized notification
            await self.send_notification(MCPMethods.INITIALIZED)

            self.initialized = True
            logger.info(f"MCP server '{self.config.name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MCP server '{self.config.name}': {e}")
            raise

    async def send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a notification (request without id) to the server."""
        if not self.process or self.process.returncode is not None:
            raise RuntimeError(f"MCP server '{self.config.name}' is not running")

        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params

        notification_json = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_json.encode())
        await self.process.stdin.drain()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        if not self.initialized:
            raise RuntimeError(f"MCP server '{self.config.name}' is not initialized")

        result = await self.send_request(MCPMethods.TOOLS_LIST)
        return result.get("tools", [])

    async def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a tool on the server."""
        if not self.initialized:
            raise RuntimeError(f"MCP server '{self.config.name}' is not initialized")

        params = {"name": name}
        if arguments:
            params["arguments"] = arguments

        result = await self.send_request(MCPMethods.TOOLS_CALL, params)
        return result

    @property
    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self.process is not None and self.process.returncode is None

    @property
    def is_initialized(self) -> bool:
        """Check if the server is initialized."""
        return self.initialized and self.is_running


class MCPServerManager:
    """Manages multiple MCP servers."""

    def __init__(self, credential_manager: Optional[CredentialManager] = None):
        self._servers: Dict[str, MCPServerProcess] = {}
        self._configs: Dict[str, MCPServerConfig] = {}
        self.credential_manager = credential_manager or (
            CredentialManager() if CredentialManager else None
        )

    async def add_server(self, config: MCPServerConfig) -> None:
        """Add and start an MCP server."""
        if config.name in self._servers:
            await self.remove_server(config.name)

        self._configs[config.name] = config
        server = MCPServerProcess(config, self.credential_manager)
        self._servers[config.name] = server

        try:
            await server.start()
            logger.info(f"Added MCP server '{config.name}'")
        except Exception as e:
            logger.error(f"Failed to add MCP server '{config.name}': {e}")
            self._servers.pop(config.name, None)
            self._configs.pop(config.name, None)
            raise

    async def remove_server(self, name: str) -> bool:
        """Remove and stop an MCP server."""
        if name not in self._servers:
            return False

        server = self._servers.pop(name)
        self._configs.pop(name, None)

        try:
            await server.stop()
            logger.info(f"Removed MCP server '{name}'")
            return True
        except Exception as e:
            logger.error(f"Error removing MCP server '{name}': {e}")
            return False

    async def get_server(self, name: str) -> Optional[MCPServerProcess]:
        """Get an MCP server by name."""
        return self._servers.get(name)

    async def list_servers(self) -> List[str]:
        """List all configured server names."""
        return list(self._servers.keys())

    async def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get tools from all servers."""
        tools = {}
        for name, server in self._servers.items():
            if server.is_initialized:
                try:
                    server_tools = await server.list_tools()
                    tools[name] = server_tools
                except Exception as e:
                    logger.error(f"Failed to get tools from server '{name}': {e}")
                    tools[name] = []
            else:
                tools[name] = []
        return tools

    async def call_tool_on_server(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a tool on a specific server."""
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"MCP server '{server_name}' not found")

        if not server.is_initialized:
            raise RuntimeError(f"MCP server '{server_name}' is not initialized")

        return await server.call_tool(tool_name, arguments)

    async def shutdown_all(self) -> None:
        """Shutdown all MCP servers."""
        logger.info("Shutting down all MCP servers")

        shutdown_tasks = []
        for server in self._servers.values():
            shutdown_tasks.append(server.stop())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self._servers.clear()
        self._configs.clear()
        logger.info("All MCP servers shut down")


# Global server manager instance
_global_manager = MCPServerManager()


async def get_global_mcp_manager() -> MCPServerManager:
    """Get the global MCP server manager."""
    return _global_manager
