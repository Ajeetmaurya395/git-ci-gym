"""
Typed Git-CI-Gym environment for repairing merge conflicts and broken CI.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        RepairCommand,
        RepairStage,
        RepoRepairAction,
        RepoRepairObservation,
        RepoRepairState,
        SourceKind,
    )
    from .tasks import TaskRegistry, TaskScenario
except ImportError:
    from models import (
        RepairCommand,
        RepairStage,
        RepoRepairAction,
        RepoRepairObservation,
        RepoRepairState,
        SourceKind,
    )
    from server.tasks import TaskRegistry, TaskScenario


class GitCIEnvironment(
    Environment[RepoRepairAction, RepoRepairObservation, RepoRepairState]
):
    """
    Real-world environment for repo repair loops.

    Each episode creates or ingests a failing repository snapshot, lets the agent
    inspect and edit files, and grades progress toward resolving merge conflicts
    and passing CI.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 18
    AVAILABLE_COMMANDS = [command.value for command in RepairCommand]
    ALLOWED_SHELL_COMMANDS = {
        "git",
        "pytest",
        "python",
        "cat",
        "ls",
        "grep",
        "find",
        "head",
        "tail",
        "diff",
        "sed",
    }
    COPY_IGNORE = shutil.ignore_patterns(
        ".git",
        "venv",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    )

    def __init__(self):
        self._workspace = tempfile.mkdtemp(prefix="gitcigym_")
        self._current_task = "easy"
        self._current_source_kind = SourceKind.builtin
        self._state = self._empty_state()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RepoRepairObservation:
        task = kwargs.get("task", "easy")
        source_kind = SourceKind(kwargs.get("source_kind", SourceKind.builtin.value))
        source_ref = kwargs.get("source_ref")
        objective = kwargs.get("objective")
        repo_label = kwargs.get("repo_label")
        source_kind, source_ref = self._normalize_source_input(source_kind, source_ref)

        self._current_task = task
        self._current_source_kind = source_kind
        self._reset_workspace()

        self._state = RepoRepairState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task,
            difficulty=task if task in TaskRegistry.list_tasks() else "custom",
            stage=RepairStage.intake,
            source_kind=source_kind,
            source_ref=source_ref or "",
            repo_label=repo_label or "",
            objective=objective or "",
            workspace_path=self._workspace,
            available_commands=list(self.AVAILABLE_COMMANDS),
        )

        try:
            if source_kind == SourceKind.builtin:
                scenario = TaskRegistry.get(task)
                self._setup_builtin_task(scenario)
                self._state.task_id = scenario.name
                self._state.difficulty = scenario.difficulty
                self._state.source_ref = source_ref or f"curriculum/{scenario.name}"
                self._state.repo_label = repo_label or f"{scenario.name.title()} Curriculum Repo"
                self._state.objective = objective or scenario.description
            elif source_kind == SourceKind.repo_path:
                self._ingest_repo_path(source_ref)
                self._state.task_id = "custom"
                self._state.difficulty = "custom"
                self._state.source_ref = source_ref or ""
                self._state.repo_label = repo_label or os.path.basename(source_ref or "local-repo")
                self._state.objective = objective or (
                    "Repair the failing local repository until merge conflicts are resolved "
                    "and CI passes."
                )
            elif source_kind == SourceKind.repo_url:
                self._ingest_repo_url(source_ref)
                self._state.task_id = "custom"
                self._state.difficulty = "custom"
                self._state.source_ref = source_ref or ""
                self._state.repo_label = repo_label or (source_ref or "remote-repo")
                self._state.objective = objective or (
                    "Repair the ingested repository until merge conflicts are resolved "
                    "and CI passes."
                )
            else:
                self._state.task_id = "custom"
                self._state.difficulty = "custom"
                self._state.source_ref = source_ref or ""
                self._state.repo_label = repo_label or "site-input"
                self._state.objective = objective or (
                    "Site-based intake is tracked here, but this environment currently "
                    "executes repo repair workflows."
                )
                self._state.notes = (
                    "Use `source_kind=repo_path` or `source_kind=repo_url` for executable inputs."
                )
        except Exception as exc:
            self._state.stage = RepairStage.failed
            self._state.last_command = "reset"
            self._state.last_result = str(exc)
            self._state.notes = str(exc)
            return self._build_observation(
                reward=0.0,
                done=True,
                last_command="reset",
                last_result=str(exc),
            )

        self._refresh_status()
        self._state.initial_failing_test_count = max(1, self._state.failing_test_count)
        self._state.last_command = "reset"
        self._state.last_result = (
            f"Workspace ready for {self._state.repo_label}. "
            f"Objective: {self._state.objective}"
        )

        return self._build_observation(
            reward=0.0,
            done=self._state.grader_score >= 1.0,
            last_command="reset",
            last_result=self._state.last_result,
        )

    def step(
        self,
        action: RepoRepairAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RepoRepairObservation:
        previous_score = self._state.grader_score
        self._state.step_count += 1

        last_result = ""
        failed_command = False
        refresh_status = action.command in {
            RepairCommand.write_file,
            RepairCommand.run_command,
            RepairCommand.get_status,
            RepairCommand.submit,
        }

        try:
            if action.command == RepairCommand.list_files:
                last_result = "\n".join(self._list_files()) or "(empty workspace)"
            elif action.command == RepairCommand.read_file:
                last_result = self._read_file(action.path or "")
            elif action.command == RepairCommand.write_file:
                last_result = self._write_file(action.path or "", action.content or "")
            elif action.command == RepairCommand.run_command:
                returncode, output = self._run_user_command(action.shell_command or "")
                last_result = output
                if returncode != 0:
                    failed_command = True
            elif action.command == RepairCommand.get_status:
                last_result = json.dumps(self._status_snapshot(), indent=2)
            elif action.command == RepairCommand.submit:
                last_result = "Submission requested. Final grader score computed from current repo state."
                refresh_status = True
            else:
                failed_command = True
                last_result = f"Unknown command: {action.command}"
        except Exception as exc:
            failed_command = True
            last_result = str(exc)

        if failed_command:
            self._state.failed_command_count += 1

        if refresh_status:
            self._refresh_status()

        done = False
        if action.command == RepairCommand.submit:
            done = True
            if self._state.grader_score >= 1.0:
                self._state.stage = RepairStage.solved
            else:
                self._state.stage = RepairStage.failed
        elif self._state.grader_score >= 1.0:
            done = True
            self._state.stage = RepairStage.solved
        elif self._state.step_count >= self.MAX_STEPS:
            done = True
            self._state.stage = RepairStage.failed
            last_result = f"{last_result}\n\nEpisode ended: reached max step budget of {self.MAX_STEPS}."

        reward = round(self._state.grader_score - previous_score, 4)
        self._state.last_command = action.command.value
        self._state.last_result = self._truncate(last_result)
        if action.notes:
            self._state.notes = action.notes

        return self._build_observation(
            reward=reward,
            done=done,
            last_command=action.command.value,
            last_result=self._state.last_result,
        )

    @property
    def state(self) -> RepoRepairState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Git-CI-Gym",
            description=(
                "Repo repair environment where an agent resolves merge conflicts, "
                "repairs failing CI, and gets deterministic grader feedback."
            ),
            version="0.2.0",
            author="Ajeet Maurya",
        )

    def close(self) -> None:
        if os.path.exists(self._workspace):
            shutil.rmtree(self._workspace, ignore_errors=True)

    def _empty_state(self) -> RepoRepairState:
        return RepoRepairState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id="easy",
            difficulty="easy",
            stage=RepairStage.intake,
            source_kind=SourceKind.builtin,
            source_ref="",
            repo_label="",
            objective="",
            workspace_path=self._workspace,
            available_commands=list(self.AVAILABLE_COMMANDS),
        )

    def _reset_workspace(self) -> None:
        if os.path.exists(self._workspace):
            shutil.rmtree(self._workspace)
        os.makedirs(self._workspace, exist_ok=True)

    def _setup_builtin_task(self, scenario: TaskScenario) -> None:
        ws = self._workspace
        self._run_internal(["git", "init", "-b", "main"], cwd=ws)
        self._run_internal(["git", "config", "user.email", "agent@gitcigym.ai"], cwd=ws)
        self._run_internal(["git", "config", "user.name", "Git-CI-Gym"], cwd=ws)

        os.makedirs(os.path.join(ws, "app", "tests"), exist_ok=True)
        open(os.path.join(ws, "app", "__init__.py"), "w").close()
        open(os.path.join(ws, "app", "tests", "__init__.py"), "w").close()

        with open(os.path.join(ws, "conftest.py"), "w") as file:
            file.write("import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n")

        with open(os.path.join(ws, "app", "main.py"), "w") as file:
            file.write(scenario.base_main_py)
        with open(os.path.join(ws, "app", "tests", "test_main.py"), "w") as file:
            file.write(scenario.base_test_py)

        self._run_internal(["git", "add", "."], cwd=ws)
        self._run_internal(["git", "commit", "-m", "Initial commit"], cwd=ws)

        self._run_internal(["git", "checkout", "-b", "feature-branch"], cwd=ws)
        with open(os.path.join(ws, "app", "main.py"), "w") as file:
            file.write(scenario.feature_main_py)
        with open(os.path.join(ws, "app", "tests", "test_main.py"), "w") as file:
            file.write(scenario.final_test_py)
        self._run_internal(["git", "add", "."], cwd=ws)
        self._run_internal(["git", "commit", "-m", "Feature branch changes"], cwd=ws)

        self._run_internal(["git", "checkout", "main"], cwd=ws)
        with open(os.path.join(ws, "app", "main.py"), "w") as file:
            file.write(scenario.main_branch_main_py)
        self._run_internal(["git", "add", "."], cwd=ws)
        self._run_internal(["git", "commit", "-m", "Main branch conflicting changes"], cwd=ws)
        self._run_internal(["git", "merge", "feature-branch", "--no-edit"], cwd=ws, check=False)

    def _ingest_repo_path(self, source_ref: str | None) -> None:
        if not source_ref or not os.path.isdir(source_ref):
            raise ValueError("`source_ref` must point to an existing local repo path.")
        shutil.copytree(source_ref, self._workspace, dirs_exist_ok=True, ignore=self.COPY_IGNORE)

    def _ingest_repo_url(self, source_ref: str | None) -> None:
        if not source_ref:
            raise ValueError("`source_ref` is required for source_kind=repo_url.")
        remote_spec = self._build_remote_spec(source_ref)
        shutil.rmtree(self._workspace, ignore_errors=True)
        parent_dir = os.path.dirname(os.path.abspath(self._workspace))
        os.makedirs(parent_dir, exist_ok=True)
        returncode, output = self._run_internal(
            ["git", "clone", "--depth", "1", remote_spec["clone_url"], self._workspace],
            cwd=parent_dir,
            check=False,
        )
        if returncode != 0:
            os.makedirs(self._workspace, exist_ok=True)
            raise ValueError(f"Failed to clone repository: {output}")
        if remote_spec.get("pull_request"):
            pr_number = remote_spec["pull_request"]
            branch_name = f"pr-{pr_number}"
            fetch_code, fetch_output = self._run_internal(
                ["git", "fetch", "origin", f"pull/{pr_number}/head:{branch_name}"],
                cwd=self._workspace,
                check=False,
            )
            if fetch_code != 0:
                raise ValueError(f"Failed to fetch GitHub pull request #{pr_number}: {fetch_output}")
            checkout_code, checkout_output = self._run_internal(
                ["git", "checkout", branch_name],
                cwd=self._workspace,
                check=False,
            )
            if checkout_code != 0:
                raise ValueError(
                    f"Fetched GitHub pull request #{pr_number} but could not check it out: "
                    f"{checkout_output}"
                )

    def _normalize_source_input(
        self, source_kind: SourceKind, source_ref: str | None
    ) -> tuple[SourceKind, str | None]:
        if source_ref:
            source_ref = source_ref.strip()
        if source_kind == SourceKind.repo_path and self._looks_like_url(source_ref):
            return SourceKind.repo_url, source_ref
        return source_kind, source_ref

    def _looks_like_url(self, source_ref: str | None) -> bool:
        if not source_ref:
            return False
        parsed = urlparse(source_ref)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _build_remote_spec(self, source_ref: str) -> dict[str, str]:
        if not self._looks_like_url(source_ref):
            return {"clone_url": source_ref}

        parsed = urlparse(source_ref)
        path_parts = [part for part in parsed.path.split("/") if part]
        if parsed.netloc != "github.com" or len(path_parts) < 2:
            return {"clone_url": source_ref}

        owner, repo = path_parts[0], path_parts[1].removesuffix(".git")
        clone_url = f"https://github.com/{owner}/{repo}.git"
        spec = {"clone_url": clone_url}
        if len(path_parts) >= 4 and path_parts[2] == "pull" and path_parts[3].isdigit():
            spec["pull_request"] = path_parts[3]
        return spec

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        python_bin = os.path.dirname(sys.executable)
        env["PATH"] = f"{python_bin}{os.pathsep}{env.get('PATH', '')}"
        return env

    def _run_internal(
        self,
        argv: list[str],
        cwd: str | None = None,
        check: bool = True,
        timeout: int = 30,
    ) -> tuple[int, str]:
        result = subprocess.run(
            argv,
            cwd=cwd or self._workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._subprocess_env(),
        )
        output = (result.stdout or "") + (result.stderr or "")
        if check and result.returncode != 0:
            raise RuntimeError(output.strip() or f"Command failed: {' '.join(argv)}")
        return result.returncode, output.strip()

    def _run_user_command(self, shell_command: str) -> tuple[int, str]:
        if not shell_command.strip():
            raise ValueError("`shell_command` cannot be empty.")
        argv = shlex.split(shell_command)
        base_command = os.path.basename(argv[0])
        if base_command not in self.ALLOWED_SHELL_COMMANDS:
            raise ValueError(
                f"Command '{base_command}' is not allowed. Allowed commands: "
                f"{sorted(self.ALLOWED_SHELL_COMMANDS)}"
            )
        if base_command == "python":
            argv[0] = sys.executable
        elif base_command == "pytest":
            argv = [sys.executable, "-m", "pytest", *argv[1:]]

        result = subprocess.run(
            argv,
            cwd=self._workspace,
            capture_output=True,
            text=True,
            timeout=30,
            env=self._subprocess_env(),
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, self._truncate(output or f"(exit code {result.returncode})")

    def _resolve_path(self, path: str) -> str:
        if not path:
            raise ValueError("A workspace-relative `path` is required.")
        full_path = os.path.abspath(os.path.join(self._workspace, path))
        workspace_root = os.path.abspath(self._workspace)
        if not full_path.startswith(workspace_root):
            raise ValueError("Path must stay inside the workspace.")
        return full_path

    def _read_file(self, path: str) -> str:
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(full_path, "r") as file:
            return self._truncate(file.read(), limit=8000)

    def _write_file(self, path: str, content: str) -> str:
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as file:
            file.write(content)
        return f"Wrote {len(content)} characters to {path}"

    def _list_files(self) -> list[str]:
        files = []
        for root, dirs, filenames in os.walk(self._workspace):
            dirs[:] = [directory for directory in dirs if directory not in {".git", "__pycache__"}]
            for name in filenames:
                rel_path = os.path.relpath(os.path.join(root, name), self._workspace)
                files.append(rel_path)
        return sorted(files)

    def _find_conflict_files(self) -> list[str]:
        conflict_files = []
        for path in self._list_files():
            full_path = os.path.join(self._workspace, path)
            try:
                with open(full_path, "r") as file:
                    content = file.read()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            if re.search(r"<{7}|={7}|>{7}", content):
                conflict_files.append(path)
        return conflict_files

    def _run_pytest(self) -> tuple[bool, int, list[str], str]:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no"],
            cwd=self._workspace,
            capture_output=True,
            text=True,
            timeout=20,
            env=self._subprocess_env(),
        )
        output = (result.stdout or "") + (result.stderr or "")
        failing_tests = []
        for line in output.splitlines():
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                parts = line.split()
                if len(parts) > 1:
                    failing_tests.append(parts[1])

        failure_count = 0
        summary_match = re.search(r"=+ (.+?) in [0-9.]+s =+", output)
        if summary_match:
            summary = summary_match.group(1)
            for number, label in re.findall(r"(\d+)\s+(failed|error)", summary):
                failure_count += int(number)
        if result.returncode != 0 and failure_count == 0:
            failure_count = len(failing_tests) or 1

        deduped_failures = list(dict.fromkeys(failing_tests))
        return result.returncode == 0, failure_count, deduped_failures, self._truncate(output, 4000)

    def _refresh_status(self) -> None:
        conflict_files = self._find_conflict_files()
        merge_resolved = len(conflict_files) == 0
        ci_passing, failing_test_count, failing_tests, pytest_output = self._run_pytest()
        score, breakdown = self._grade_progress(
            merge_resolved=merge_resolved,
            ci_passing=ci_passing,
            failing_test_count=failing_test_count,
        )

        self._state.conflict_files = conflict_files
        self._state.merge_resolved = merge_resolved
        self._state.ci_passing = ci_passing
        self._state.failing_test_count = failing_test_count
        self._state.failing_tests = failing_tests
        self._state.grader_score = score
        self._state.grader_breakdown = breakdown
        self._state.workspace_snapshot = self._list_files()[:20]

        if score >= 1.0:
            self._state.stage = RepairStage.solved
        elif not merge_resolved:
            self._state.stage = RepairStage.repair
        elif not ci_passing:
            self._state.stage = RepairStage.verify
        else:
            self._state.stage = RepairStage.analyze

        if not ci_passing and not self._state.notes:
            self._state.notes = pytest_output

    def _grade_progress(
        self,
        merge_resolved: bool,
        ci_passing: bool,
        failing_test_count: int,
    ) -> tuple[float, dict[str, float]]:
        initial_failures = max(1, self._state.initial_failing_test_count or failing_test_count or 1)
        merge_score = 0.35 if merge_resolved else 0.0
        ci_progress = 0.35 * max(0.0, 1.0 - (failing_test_count / initial_failures))
        ci_pass_bonus = 0.30 if ci_passing else 0.0
        failed_command_penalty = min(0.20, 0.04 * self._state.failed_command_count)
        total = max(0.0, min(1.0, merge_score + ci_progress + ci_pass_bonus - failed_command_penalty))
        breakdown = {
            "merge_resolution": round(merge_score, 4),
            "ci_progress": round(ci_progress, 4),
            "ci_pass_bonus": round(ci_pass_bonus, 4),
            "failed_command_penalty": round(-failed_command_penalty, 4),
        }
        return round(total, 4), breakdown

    def _status_snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self._state.task_id,
            "difficulty": self._state.difficulty,
            "stage": self._state.stage.value,
            "source_kind": self._state.source_kind.value,
            "source_ref": self._state.source_ref,
            "repo_label": self._state.repo_label,
            "objective": self._state.objective,
            "workspace_path": self._state.workspace_path,
            "conflict_files": self._state.conflict_files,
            "failing_tests": self._state.failing_tests,
            "failing_test_count": self._state.failing_test_count,
            "merge_resolved": self._state.merge_resolved,
            "ci_passing": self._state.ci_passing,
            "grader_score": self._state.grader_score,
            "grader_breakdown": self._state.grader_breakdown,
            "failed_command_count": self._state.failed_command_count,
            "step_count": self._state.step_count,
        }

    def _build_observation(
        self,
        reward: float,
        done: bool,
        last_command: str,
        last_result: str,
    ) -> RepoRepairObservation:
        return RepoRepairObservation(
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            stage=self._state.stage,
            source_kind=self._state.source_kind,
            source_ref=self._state.source_ref,
            repo_label=self._state.repo_label,
            objective=self._state.objective,
            available_commands=list(self._state.available_commands),
            workspace_path=self._state.workspace_path,
            workspace_snapshot=list(self._state.workspace_snapshot),
            conflict_files=list(self._state.conflict_files),
            failing_tests=list(self._state.failing_tests),
            failing_test_count=self._state.failing_test_count,
            merge_resolved=self._state.merge_resolved,
            ci_passing=self._state.ci_passing,
            grader_score=self._state.grader_score,
            grader_breakdown=dict(self._state.grader_breakdown),
            last_command=last_command,
            last_result=self._truncate(last_result),
            notes=self._state.notes,
            reward=reward,
            done=done,
            metadata={
                "step_count": self._state.step_count,
                "failed_command_count": self._state.failed_command_count,
            },
        )

    def _truncate(self, text: str, limit: int = 2400) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n... [truncated]"
