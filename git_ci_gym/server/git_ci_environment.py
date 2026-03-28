"""
Git-CI-Gym Environment Implementation.

An MCP environment where an AI agent resolves Git merge conflicts
and fixes resulting CI/CD test failures (pytest).

Tools exposed via FastMCP:
- `edit_file(path, content)`: Write content to a file in the workspace.
- `read_file(path)`: Read a file from the workspace.
- `run_command(command)`: Execute a shell command (git, pytest, etc.).
- `list_files()`: List all files in the workspace.
- `get_status()`: Get current merge/CI status summary.
"""

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

from .tasks import TaskRegistry


class GitCIEnvironment(MCPEnvironment):
    """
    Git-CI-Gym: An RL environment for semantic code merging and CI repair.

    The agent is dropped into a git repository with an active merge conflict.
    It must:
    1. Resolve the conflict markers (<<<<<<< / ======= / >>>>>>>)
    2. Fix any resulting test failures so pytest passes

    Reward shaping:
    - +0.3 for resolving all merge markers (git merge --continue succeeds)
    - +0.7 for all pytest tests passing (exit code 0)
    - -0.05 penalty for each failed pytest run
    """

    ALLOWED_COMMANDS = ["git", "pytest", "python", "cat", "ls", "grep", "find", "head", "tail", "diff"]

    def __init__(self):
        """Initialize the Git-CI-Gym environment with MCP tools."""
        mcp = FastMCP("git_ci_gym")
        self._workspace = tempfile.mkdtemp(prefix="gitcigym_")
        self._current_task = "easy"
        self._failed_ci_runs = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # ── Define MCP Tools ─────────────────────────────────────────

        @mcp.tool
        def edit_file(path: str, content: str) -> str:
            """
            Write content to a file in the workspace.

            Args:
                path: Relative path within the workspace (e.g., 'app/main.py').
                content: The full new content for the file.

            Returns:
                Confirmation message or error.
            """
            full_path = os.path.join(self._workspace, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            return f"✅ File '{path}' written ({len(content)} chars)."

        @mcp.tool
        def read_file(path: str) -> str:
            """
            Read a file from the workspace.

            Args:
                path: Relative path within the workspace.

            Returns:
                The file content, or an error message.
            """
            full_path = os.path.join(self._workspace, path)
            if not os.path.exists(full_path):
                return f"❌ File not found: {path}"
            with open(full_path, "r") as f:
                return f.read()

        @mcp.tool
        def run_command(command: str) -> str:
            """
            Execute a shell command in the workspace.
            Allowed commands: git, pytest, python, cat, ls, grep, find, head, tail, diff.

            Args:
                command: The shell command to run.

            Returns:
                Combined stdout + stderr output.
            """
            # Security: validate only allowed executables
            cmd_parts = command.strip().split()
            if not cmd_parts:
                return "❌ Empty command."
            base_cmd = os.path.basename(cmd_parts[0])
            if base_cmd not in self.ALLOWED_COMMANDS:
                return f"❌ Command '{base_cmd}' is not allowed. Allowed: {self.ALLOWED_COMMANDS}"

            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=self._workspace,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = result.stdout + result.stderr

                # Track failed CI runs for reward penalty
                if base_cmd == "pytest" and result.returncode != 0:
                    self._failed_ci_runs += 1

                return output if output.strip() else f"(exit code {result.returncode})"
            except subprocess.TimeoutExpired:
                return "❌ Command timed out (30s limit)."
            except Exception as e:
                return f"❌ Error: {str(e)}"

        @mcp.tool
        def list_files() -> str:
            """
            List all files in the workspace recursively.

            Returns:
                Newline-separated list of file paths.
            """
            files = []
            for root, dirs, filenames in os.walk(self._workspace):
                # Skip .git internals
                dirs[:] = [d for d in dirs if d != ".git"]
                for fname in filenames:
                    rel = os.path.relpath(os.path.join(root, fname), self._workspace)
                    files.append(rel)
            return "\n".join(sorted(files)) if files else "(empty workspace)"

        @mcp.tool
        def get_status() -> dict:
            """
            Get the current status of the merge and CI.

            Returns:
                Dictionary with merge_resolved, ci_passing, conflict_files,
                failed_ci_runs, step_count, and reward_breakdown.
            """
            merge_resolved = self._is_merge_resolved()
            ci_passing = self._is_ci_passing()

            # Find conflict files
            conflict_files = self._get_conflict_files()

            # Calculate reward
            reward_breakdown = {
                "marker_removal": 0.3 if merge_resolved else 0.0,
                "ci_pass": 0.7 if ci_passing else 0.0,
                "penalty": round(-0.05 * self._failed_ci_runs, 2),
            }
            total = sum(reward_breakdown.values())

            return {
                "merge_resolved": merge_resolved,
                "ci_passing": ci_passing,
                "conflict_files": conflict_files,
                "failed_ci_runs": self._failed_ci_runs,
                "step_count": self._state.step_count,
                "reward": round(max(-1.0, min(1.0, total)), 2),
                "reward_breakdown": reward_breakdown,
                "task": self._current_task,
            }

        # Pass mcp to base class
        super().__init__(mcp)

    # ── OpenEnv Interface ────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment to a fresh git conflict state.

        Kwargs:
            task: Difficulty level — 'easy', 'medium', or 'hard'. Default: 'easy'.
        """
        task = kwargs.get("task", "easy")
        self._current_task = task
        self._failed_ci_runs = 0
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Clean workspace
        if os.path.exists(self._workspace):
            shutil.rmtree(self._workspace)
        os.makedirs(self._workspace)

        scenario = TaskRegistry.get(task)
        self._setup_git_conflict(scenario)

        # Build initial status
        conflict_files = self._get_conflict_files()

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "conflict",
                "task": task,
                "difficulty": scenario.difficulty,
                "description": scenario.description,
                "conflict_files": conflict_files,
                "message": (
                    f"🔥 Git-CI-Gym [{task.upper()}]: Merge conflict detected! "
                    f"Conflicting files: {conflict_files}. "
                    "Use the tools to resolve the conflict and make all tests pass."
                ),
                "available_tools": [
                    "edit_file(path, content)",
                    "read_file(path)",
                    "run_command(command)",
                    "list_files()",
                    "get_status()",
                ],
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (returns error — use tools instead)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use MCP tools: edit_file, read_file, run_command, list_files, get_status."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step. Increments step count, then delegates to MCPEnvironment."""
        self._state.step_count += 1

        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Check if done (merge resolved + CI green)
        merge_ok = self._is_merge_resolved()
        ci_ok = self._is_ci_passing()
        done = merge_ok and ci_ok

        if done:
            reward_breakdown = {
                "marker_removal": 0.3,
                "ci_pass": 0.7,
                "penalty": round(-0.05 * self._failed_ci_runs, 2),
            }
            total = round(max(-1.0, min(1.0, sum(reward_breakdown.values()))), 2)
            return Observation(
                done=True,
                reward=total,
                metadata={
                    **obs.metadata,
                    "message": f"🎉 Task '{self._current_task}' SOLVED! Final reward: {total}",
                    "reward_breakdown": reward_breakdown,
                },
            )

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step for WebSocket handler."""
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    # ── Internal Helpers ─────────────────────────────────────────────

    def _run(self, cmd: str) -> str:
        """Run a shell command in the workspace."""
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=self._workspace,
                capture_output=True, text=True, timeout=15,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return str(e)

    def _setup_git_conflict(self, scenario) -> None:
        """Set up a git repo with a merge conflict based on the scenario."""
        ws = self._workspace

        # Init repo with explicit main branch name
        self._run("git init -b main")
        self._run('git config user.email "agent@gitcigym.ai"')
        self._run('git config user.name "Git-CI-Gym"')

        # Create app directories
        os.makedirs(os.path.join(ws, "app", "tests"), exist_ok=True)

        # Write __init__.py files
        open(os.path.join(ws, "app", "__init__.py"), "w").close()
        open(os.path.join(ws, "app", "tests", "__init__.py"), "w").close()

        # Write conftest for pytest to find app module
        with open(os.path.join(ws, "conftest.py"), "w") as f:
            f.write("import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n")

        # ── Base commit on main ──────────────────────────────────────
        with open(os.path.join(ws, "app", "main.py"), "w") as f:
            f.write(scenario.base_main_py)
        with open(os.path.join(ws, "app", "tests", "test_main.py"), "w") as f:
            f.write(scenario.base_test_py)

        self._run("git add .")
        self._run('git commit -m "Initial commit"')

        # ── Feature branch: change main.py AND test file ─────────────
        self._run("git checkout -b feature-branch")
        with open(os.path.join(ws, "app", "main.py"), "w") as f:
            f.write(scenario.feature_main_py)
        with open(os.path.join(ws, "app", "tests", "test_main.py"), "w") as f:
            f.write(scenario.final_test_py)
        self._run("git add .")
        self._run('git commit -m "Feature branch changes"')

        # ── Back to main: create CONFLICTING change to same lines ────
        self._run("git checkout main")
        with open(os.path.join(ws, "app", "main.py"), "w") as f:
            f.write(scenario.main_branch_main_py)
        self._run("git add .")
        self._run('git commit -m "Main branch conflicting changes"')

        # ── Trigger merge conflict ───────────────────────────────────
        merge_output = self._run("git merge feature-branch --no-edit")

        # Verify we're on main and conflict exists
        branch_output = self._run("git branch --show-current").strip()
        if branch_output != "main":
            # Fallback: try to get back to main
            self._run("git checkout main")

        # Log for debugging
        self._merge_output = merge_output

    def _get_conflict_files(self) -> list:
        """Get list of files with merge conflict markers."""
        output = self._run("git diff --name-only --diff-filter=U")
        return [f.strip() for f in output.strip().split("\n") if f.strip()]

    def _has_markers(self) -> bool:
        """Check if any file in the workspace still has merge markers."""
        for root, dirs, files in os.walk(self._workspace):
            dirs[:] = [d for d in dirs if d != ".git"]
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r") as f:
                        content = f.read()
                    if re.search(r"<{7}|={7}|>{7}", content):
                        return True
                except (UnicodeDecodeError, PermissionError):
                    continue
        return False

    def _is_merge_resolved(self) -> bool:
        """Check if merge is resolved (no conflict markers remain)."""
        return not self._has_markers()

    def _is_ci_passing(self) -> bool:
        """Run pytest and check if all tests pass."""
        try:
            result = subprocess.run(
                "pytest app/tests/ -q --tb=no",
                shell=True, cwd=self._workspace,
                capture_output=True, text=True, timeout=15,
            )
            return result.returncode == 0
        except Exception:
            return False
