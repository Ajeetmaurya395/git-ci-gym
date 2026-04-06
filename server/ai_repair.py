"""
AI Repair Agent for Git-CI-Gym.

Implements a Sense-Think-Act agentic loop that:
  1. Lists files to understand workspace layout
  2. Reads conflicted files to see merge markers
  3. Analyzes the conflict, tests, and objective
  4. Writes resolved content
  5. Runs pytest to verify
  6. Iterates until grader score reaches 1.0

Uses Gemini (via OpenAI-compatible endpoint) for speed and cost-efficiency.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Generator

from openai import OpenAI

try:
    from ..models import RepairCommand, RepoRepairAction, RepoRepairObservation
    from .git_ci_environment import GitCIEnvironment
    from .tasks import TaskRegistry
    from .github_intel import PRIntelligence
except ImportError:
    from models import RepairCommand, RepoRepairAction, RepoRepairObservation
    from server.git_ci_environment import GitCIEnvironment
    from server.tasks import TaskRegistry
    from server.github_intel import PRIntelligence


REPAIR_SYSTEM_PROMPT = """\
You are an expert AI developer performing a repository repair.

You are inside a Git-CI-Gym episode. Your mission is to:
1. Inspect the repository structure
2. Find and read files with merge conflict markers (<<<<<<< / ======= / >>>>>>>)
3. Resolve conflicts by understanding BOTH sides and the test expectations
4. Write the corrected file content
5. Run tests to verify your fix
6. Iterate until ALL tests pass

## Rules:
- ALWAYS respond with valid JSON only. No markdown, no explanation outside JSON.
- Use exactly ONE of these action shapes per response:

  {"command": "list_files"}
  {"command": "read_file", "path": "relative/path"}
  {"command": "edit_file", "path": "relative/path", "search_term": "exact old text to replace", "replacement": "new replacement text"}
  {"command": "run_command", "shell_command": "pytest -q --tb=short"}
  {"command": "get_status"}
  {"command": "submit"}

## Strategy:
- Start with list_files to see the workspace
- Read each conflicted file to understand both sides of the conflict
- Read the test file to understand what the resolved code must do
- Check the PR Intelligence context (if provided) to understand the failure details
- Use edit_file to surgically update blocks of code instead of writing full files
- When editing, ensure `search_term` is an exact substring match (including whitespace)
- Run pytest to check. If tests fail, read the traceback, fix, and retry.
- When grader_score reaches 1.0, submit.

## Conflict Resolution Guidelines:
- NEVER leave <<<<<<< , ======= , or >>>>>>> markers in the file
- Understand what each branch was trying to do
- Merge semantics: keep BOTH features when possible
- Make sure the code matches what the tests expect
- If a function was renamed, update ALL references

## Important:
- You have a limited step budget. Be efficient.
- After writing a fix, ALWAYS run pytest to verify.
- Only submit when you are confident the score is 1.0.
"""


@dataclass
class RepairStep:
    """One step in the repair process, used for streaming to the UI."""

    step_number: int
    command: str
    detail: str = ""
    result_preview: str = ""
    grader_score: float = 0.0
    stage: str = "repair"
    done: bool = False
    error: str = ""
    merge_resolved: bool = False
    ci_passing: bool = False
    grader_breakdown: dict[str, float] = field(default_factory=dict)
    conflict_files: list[str] = field(default_factory=list)
    failing_tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_number,
            "command": self.command,
            "detail": self.detail,
            "result_preview": self.result_preview[:600],
            "grader_score": self.grader_score,
            "stage": self.stage,
            "done": self.done,
            "error": self.error,
            "merge_resolved": self.merge_resolved,
            "ci_passing": self.ci_passing,
            "grader_breakdown": self.grader_breakdown,
            "conflict_files": self.conflict_files,
            "failing_tests": self.failing_tests,
        }

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_dict())}\n\n"


def _create_client() -> OpenAI | None:
    """Create an OpenAI client, prioritizing robust free tiers like Gemini and Groq."""
    
    # Priority 1: Groq native
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )

    # Priority 2: Google Gemini native
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        
    # Priority 3: Standard OpenAI (or OpenRouter/LiteLLM proxy which may hit 429 limits)
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("API_BASE_URL"):
        return OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY") or "dummy-key-for-proxy",
            base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
        )
        
    return None


def _model_name() -> str:
    return os.environ.get("REPAIR_MODEL", "gemini-2.0-flash")


def _observation_to_context(obs: RepoRepairObservation) -> str:
    """Convert an observation to a concise context string for the LLM."""
    return json.dumps(
        {
            "stage": obs.stage.value if hasattr(obs.stage, "value") else str(obs.stage),
            "objective": obs.objective,
            "conflict_files": obs.conflict_files,
            "failing_tests": obs.failing_tests,
            "failing_test_count": obs.failing_test_count,
            "merge_resolved": obs.merge_resolved,
            "ci_passing": obs.ci_passing,
            "grader_score": obs.grader_score,
            "grader_breakdown": obs.grader_breakdown,
            "workspace_snapshot": obs.workspace_snapshot[:15],
            "last_command": obs.last_command,
            "last_result": obs.last_result[:1500] if obs.last_result else "",
        },
        indent=2,
    )


def _parse_llm_action(content: str) -> dict[str, Any]:
    """Extract a JSON action from possibly messy LLM output."""
    # Try to find JSON block
    content = content.strip()
    # Remove markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        raise ValueError(f"No JSON found in LLM response: {content[:200]}")
    return json.loads(content[json_start:json_end])


def _heuristic_action(
    task_name: str,
    observation: RepoRepairObservation,
    step: int,
) -> RepoRepairAction:
    """Scripted fallback for when the LLM is unavailable.
    
    For builtin tasks: applies the known solution directly.
    For custom repos: does a systematic inspect-and-submit flow.
    """
    # Step 1: always list files first
    if step == 1:
        return RepoRepairAction(
            command=RepairCommand.list_files,
            notes="Scripted: listing workspace files",
        )

    # For builtin tasks with known solutions — apply directly
    if observation.conflict_files:
        try:
            scenario = TaskRegistry.get(task_name)
            return RepoRepairAction(
                command=RepairCommand.write_file,
                path="app/main.py",
                content=scenario.solution_main_py,
                notes="Scripted: applying known solution for builtin task",
            )
        except (ValueError, KeyError):
            pass

    # Read conflict files
    if observation.conflict_files:
        cf_idx = step - 2 
        if 0 <= cf_idx < len(observation.conflict_files):
            return RepoRepairAction(
                command=RepairCommand.read_file,
                path=observation.conflict_files[cf_idx],
                notes=f"Scripted: reading conflict file {cf_idx + 1}/{len(observation.conflict_files)}",
            )

    # Read failing CI tests 
    if observation.failing_tests:
        # offset by conflict files checked
        offset_step = step - 2 - len(observation.conflict_files)
        if 0 <= offset_step < len(observation.failing_tests):
            return RepoRepairAction(
                command=RepairCommand.read_file,
                path=observation.failing_tests[offset_step],
                notes=f"Scripted: analyzing failing CI test {offset_step + 1}/{len(observation.failing_tests)}",
            )

    # Run get_status once to understand the state
    if step <= 5 + max(len(observation.conflict_files), len(observation.failing_tests)):
        return RepoRepairAction(
            command=RepairCommand.get_status,
            notes="Scripted: checking current status",
        )

    # Submit — don't waste remaining steps looping on commands
    return RepoRepairAction(
        command=RepairCommand.submit,
        notes="Scripted: submitting (LLM required for complex auto-repairs)",
    )

def run_repair_agent(
    env: GitCIEnvironment,
    observation: RepoRepairObservation,
    task_name: str = "custom",
    max_steps: int = 12,
    pr_intel: PRIntelligence | None = None,
) -> Generator[RepairStep, None, None]:
    """
    Run the repair agent loop, yielding RepairStep objects for SSE streaming.

    This is the core agentic loop:
      Sense (observe) → Think (LLM decides action) → Act (execute in env)
    
    If pr_intel is provided, the agent receives PR context (CI failures,
    diff, review comments, security scan) for more targeted repairs.
    """
    client = _create_client()
    model = _model_name()
    use_llm = client is not None
    llm_failed_permanently = False

    # Build system prompt — enhanced with PR intel if available
    system_prompt = REPAIR_SYSTEM_PROMPT
    if pr_intel and pr_intel.fetched:
        pr_context_str = pr_intel.to_agent_context()
        system_prompt += (
            "\n\n## PR Intelligence (from GitHub API)\n"
            "Use this context to understand what the PR is trying to do, "
            "what CI checks are failing and why, and what reviewers have said.\n\n"
            f"{pr_context_str}\n"
        )

    # Build initial transcript
    transcript: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Task: {task_name}\n"
                f"Objective: {observation.objective}\n\n"
                f"Current state:\n{_observation_to_context(observation)}\n\n"
                "Choose the best next action. Respond with JSON only."
            ),
        },
    ]

    current_obs = observation
    heuristic_step = 0  # separate counter for heuristic sequencing

    for step_num in range(1, max_steps + 1):
        if current_obs.done:
            yield RepairStep(
                step_number=step_num,
                command="done",
                detail="Episode complete",
                grader_score=current_obs.grader_score,
                stage=current_obs.stage.value if hasattr(current_obs.stage, "value") else str(current_obs.stage),
                done=True,
                merge_resolved=current_obs.merge_resolved,
                ci_passing=current_obs.ci_passing,
                grader_breakdown=dict(current_obs.grader_breakdown),
                conflict_files=list(current_obs.conflict_files),
                failing_tests=list(current_obs.failing_tests),
            )
            return

        # ── THINK ──────────────────────────────────────────────
        action_dict = None
        try:
            if use_llm and not llm_failed_permanently:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=transcript,
                    max_tokens=4096,
                )
                raw_content = response.choices[0].message.content or "{}"
                action_dict = _parse_llm_action(raw_content)
                action = RepoRepairAction.model_validate(action_dict)
            else:
                heuristic_step += 1
                action = _heuristic_action(task_name, current_obs, heuristic_step)
        except Exception as exc:
            # LLM failed — fall back to heuristic for this step AND all future steps
            llm_failed_permanently = True
            heuristic_step += 1
            yield RepairStep(
                step_number=step_num,
                command="fallback",
                detail=f"LLM unavailable ({type(exc).__name__}), switching to heuristic solver",
                error=str(exc)[:200],
                grader_score=current_obs.grader_score,
                stage=current_obs.stage.value if hasattr(current_obs.stage, "value") else str(current_obs.stage),
                merge_resolved=current_obs.merge_resolved,
                ci_passing=current_obs.ci_passing,
                grader_breakdown=dict(current_obs.grader_breakdown),
                conflict_files=list(current_obs.conflict_files),
                failing_tests=list(current_obs.failing_tests),
            )
            action = _heuristic_action(task_name, current_obs, heuristic_step)

        # Build friendly detail string
        detail = action.command.value
        if action.path:
            detail += f" → {action.path}"
        if action.shell_command:
            detail += f" → {action.shell_command}"
        if action.notes:
            detail += f" ({action.notes})"

        # ── ACT ────────────────────────────────────────────────
        try:
            result_obs = env.step(action)
        except Exception as exc:
            yield RepairStep(
                step_number=step_num,
                command=action.command.value,
                detail=detail,
                error=f"Execution failed: {exc}",
                grader_score=current_obs.grader_score,
                stage=current_obs.stage.value if hasattr(current_obs.stage, "value") else str(current_obs.stage),
                merge_resolved=current_obs.merge_resolved,
                ci_passing=current_obs.ci_passing,
                grader_breakdown=dict(current_obs.grader_breakdown),
                conflict_files=list(current_obs.conflict_files),
                failing_tests=list(current_obs.failing_tests),
            )
            continue

        current_obs = result_obs

        # ── SENSE (yield step for streaming) ───────────────────
        yield RepairStep(
            step_number=step_num,
            command=action.command.value,
            detail=detail,
            result_preview=current_obs.last_result or "",
            grader_score=current_obs.grader_score,
            stage=current_obs.stage.value if hasattr(current_obs.stage, "value") else str(current_obs.stage),
            done=current_obs.done,
            merge_resolved=current_obs.merge_resolved,
            ci_passing=current_obs.ci_passing,
            grader_breakdown=dict(current_obs.grader_breakdown),
            conflict_files=list(current_obs.conflict_files),
            failing_tests=list(current_obs.failing_tests),
        )

        # Update transcript for next LLM call
        if use_llm and not llm_failed_permanently and action_dict is not None:
            transcript.append(
                {"role": "assistant", "content": json.dumps(action_dict)}
            )
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        f"Step {step_num} result:\n"
                        f"{_observation_to_context(current_obs)}\n\n"
                        "Choose the next action. Respond with JSON only."
                    ),
                }
            )

        if current_obs.done:
            return

    # Max steps exhausted
    yield RepairStep(
        step_number=max_steps + 1,
        command="timeout",
        detail=f"Reached max step budget ({max_steps})",
        grader_score=current_obs.grader_score,
        stage=current_obs.stage.value if hasattr(current_obs.stage, "value") else str(current_obs.stage),
        done=True,
        merge_resolved=current_obs.merge_resolved,
        ci_passing=current_obs.ci_passing,
        grader_breakdown=dict(current_obs.grader_breakdown),
        conflict_files=list(current_obs.conflict_files),
        failing_tests=list(current_obs.failing_tests),
    )
