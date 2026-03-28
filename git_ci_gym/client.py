"""
Git-CI-Gym Client.

Provides the client for connecting to a Git-CI-Gym server.
GitCIEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with GitCIEnv(base_url="http://localhost:7860").sync() as env:
    ...     result = env.reset(task="medium")
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...     # ['edit_file', 'read_file', 'run_command', 'list_files', 'get_status']
    ...
    ...     # Read the conflicting file
    ...     content = env.call_tool("read_file", path="app/main.py")
    ...     print(content)
    ...
    ...     # Fix the file
    ...     env.call_tool("edit_file", path="app/main.py", content="...")
    ...
    ...     # Run tests
    ...     result = env.call_tool("run_command", command="pytest app/tests/ -v")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class GitCIEnv(MCPToolClient):
    """
    Client for the Git-CI-Gym Environment.

    Inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)
    """

    pass  # MCPToolClient provides all needed functionality
