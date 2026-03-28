"""
FastAPI application for Git-CI-Gym Environment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    # Or: uv run --project . server
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .git_ci_environment import GitCIEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.git_ci_environment import GitCIEnvironment

# Create the app — pass the class (factory) for WebSocket session support
app = create_app(
    GitCIEnvironment, CallToolAction, CallToolObservation, env_name="git_ci_gym"
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
