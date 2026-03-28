"""
Git-CI-Gym Models.

Re-exports the standard OpenEnv types. Custom types are not needed
since all interaction happens through MCP tools.
"""

try:
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from openenv.core.env_server.types import Action, Observation, State

__all__ = [
    "Action",
    "Observation",
    "State",
    "CallToolAction",
    "CallToolObservation",
]
