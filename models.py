"""
Typed models for the Git-CI-Gym repo repair environment.
"""

from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, model_validator


class SourceKind(str, Enum):
    """Supported source types for a repair episode."""

    builtin = "builtin"
    repo_path = "repo_path"
    repo_url = "repo_url"
    site_url = "site_url"


class RepairStage(str, Enum):
    """High-level episode stage shown in the UI and observations."""

    intake = "intake"
    analyze = "analyze"
    repair = "repair"
    verify = "verify"
    solved = "solved"
    failed = "failed"


class RepairCommand(str, Enum):
    """Typed actions available to the agent."""

    list_files = "list_files"
    read_file = "read_file"
    write_file = "write_file"
    run_command = "run_command"
    get_status = "get_status"
    submit = "submit"


class RepoRepairAction(Action):
    """One environment action in the repo repair loop."""

    command: RepairCommand = Field(..., description="Command to execute")
    path: str | None = Field(default=None, description="Workspace-relative path")
    content: str | None = Field(default=None, description="Full file content for writes")
    shell_command: str | None = Field(
        default=None, description="Shell command for run_command"
    )
    notes: str | None = Field(
        default=None, description="Optional intent or rationale from the agent"
    )

    @model_validator(mode="after")
    def validate_required_fields(self) -> "RepoRepairAction":
        if self.command in {RepairCommand.read_file, RepairCommand.write_file} and not self.path:
            raise ValueError(f"`path` is required for {self.command.value}")
        if self.command == RepairCommand.write_file and self.content is None:
            raise ValueError("`content` is required for write_file")
        if self.command == RepairCommand.run_command and not self.shell_command:
            raise ValueError("`shell_command` is required for run_command")
        return self


class RepoRepairObservation(Observation):
    """Structured observation for repo repair episodes."""

    task_id: str = Field(default="easy", description="Current task identifier")
    difficulty: str = Field(default="easy", description="Task difficulty level")
    stage: RepairStage = Field(
        default=RepairStage.intake, description="Current stage of the workflow"
    )
    source_kind: SourceKind = Field(
        default=SourceKind.builtin, description="Where the repo input came from"
    )
    source_ref: str = Field(default="", description="Repo path, URL, or label")
    repo_label: str = Field(default="", description="Human-readable source label")
    objective: str = Field(default="", description="What the agent must accomplish")
    available_commands: list[str] = Field(
        default_factory=list, description="Commands the agent can issue next"
    )
    workspace_path: str = Field(default="", description="Temporary workspace path")
    workspace_snapshot: list[str] = Field(
        default_factory=list, description="Sample of current workspace files"
    )
    conflict_files: list[str] = Field(
        default_factory=list, description="Files that still contain merge markers"
    )
    failing_tests: list[str] = Field(
        default_factory=list, description="Tests currently failing in CI"
    )
    failing_test_count: int = Field(
        default=0, description="Number of failing tests detected by the grader"
    )
    merge_resolved: bool = Field(
        default=False, description="Whether all merge markers have been removed"
    )
    ci_passing: bool = Field(default=False, description="Whether pytest currently passes")
    grader_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Absolute grader score from 0 to 1"
    )
    grader_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Deterministic breakdown of grader components"
    )
    last_command: str = Field(default="reset", description="Most recent command")
    last_result: str = Field(default="", description="Result of the most recent command")
    notes: str = Field(default="", description="Additional execution notes")


class RepoRepairState(State):
    """Persistent environment state for repo repair episodes."""

    task_id: str = Field(default="easy")
    difficulty: str = Field(default="easy")
    stage: RepairStage = Field(default=RepairStage.intake)
    source_kind: SourceKind = Field(default=SourceKind.builtin)
    source_ref: str = Field(default="")
    repo_label: str = Field(default="")
    objective: str = Field(default="")
    workspace_path: str = Field(default="")
    workspace_snapshot: list[str] = Field(default_factory=list)
    available_commands: list[str] = Field(default_factory=list)
    conflict_files: list[str] = Field(default_factory=list)
    failing_tests: list[str] = Field(default_factory=list)
    failing_test_count: int = Field(default=0)
    initial_failing_test_count: int = Field(default=0)
    merge_resolved: bool = Field(default=False)
    ci_passing: bool = Field(default=False)
    grader_score: float = Field(default=0.0, ge=0.0, le=1.0)
    grader_breakdown: dict[str, float] = Field(default_factory=dict)
    failed_command_count: int = Field(default=0)
    last_command: str = Field(default="reset")
    last_result: str = Field(default="")
    notes: str = Field(default="")
    source_metadata: dict[str, Any] = Field(default_factory=dict)
