"""
Baseline inference for the typed Git-CI-Gym repo repair environment.

Reads the following environment variables when available:
  - API_BASE_URL
  - MODEL_NAME
  - HF_TOKEN or OPENAI_API_KEY
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

from git_ci_gym.models import RepairCommand, RepoRepairAction, RepoRepairObservation
from server.git_ci_environment import GitCIEnvironment
from server.tasks import TaskRegistry


SYSTEM_PROMPT = """You are an autonomous repo repair agent.

You are inside a Git-CI-Gym episode. Your job is to inspect the repository,
remove merge conflicts, repair broken CI, and stop when the grader score reaches 1.0.

Available commands:
- list_files
- read_file(path)
- write_file(path, content)
- run_command(shell_command)
- get_status
- submit

Always respond with valid JSON only, shaped like:
{"command":"read_file","path":"app/main.py","notes":"optional short rationale"}

When you need shell access, use:
{"command":"run_command","shell_command":"pytest -q --tb=no"}
"""


def create_openai_client() -> OpenAI | None:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    )


def model_name() -> str:
    return os.environ.get("MODEL_NAME", "gpt-4o-mini")


def observation_summary(obs: RepoRepairObservation) -> str:
    return json.dumps(
        {
            "task_id": obs.task_id,
            "difficulty": obs.difficulty,
            "stage": obs.stage.value,
            "objective": obs.objective,
            "repo_label": obs.repo_label,
            "conflict_files": obs.conflict_files,
            "failing_tests": obs.failing_tests,
            "failing_test_count": obs.failing_test_count,
            "merge_resolved": obs.merge_resolved,
            "ci_passing": obs.ci_passing,
            "grader_score": obs.grader_score,
            "workspace_snapshot": obs.workspace_snapshot,
            "last_result": obs.last_result,
            "available_commands": obs.available_commands,
        },
        indent=2,
    )


def heuristic_action(task_name: str, observation: RepoRepairObservation) -> RepoRepairAction:
    if observation.conflict_files:
        scenario = TaskRegistry.get(task_name)
        return RepoRepairAction(
            command=RepairCommand.write_file,
            path="app/main.py",
            content=scenario.solution_main_py,
            notes="Scripted curriculum fallback applies the known task solution.",
        )
    if not observation.ci_passing:
        return RepoRepairAction(
            command=RepairCommand.run_command,
            shell_command="pytest -q --tb=no",
            notes="Scripted fallback verifies the repaired repository.",
        )
    return RepoRepairAction(command=RepairCommand.submit, notes="Scripted fallback submits once green.")


def llm_action(
    client: OpenAI,
    model: str,
    task_name: str,
    transcript: list[dict[str, str]],
) -> RepoRepairAction:
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=transcript,
        max_tokens=1800,
    )
    content = response.choices[0].message.content or "{}"
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    payload = json.loads(content[json_start:json_end])
    return RepoRepairAction.model_validate(payload)


def run_task(
    env: GitCIEnvironment,
    task_name: str,
    client: OpenAI | None,
    model: str,
    max_steps: int = 10,
) -> dict[str, Any]:
    print(f"\n{'=' * 64}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'=' * 64}")

    initial = env.reset(task=task_name, source_kind="builtin")
    print(f"Objective: {initial.objective}")
    print(f"Conflicts: {initial.conflict_files}")
    print(f"Initial grader score: {initial.grader_score:.2f}")

    transcript = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {task_name}\n"
                f"Current observation:\n{observation_summary(initial)}\n\n"
                "Choose the best next command."
            ),
        },
    ]

    observation = initial
    reward = 0.0

    for step in range(1, max_steps + 1):
        if observation.done:
            break

        if client is None:
            action = heuristic_action(task_name, observation)
        else:
            action = llm_action(client, model, task_name, transcript)

        print(f"[step {step}] {action.command.value}")
        if action.path:
            print(f"  path={action.path}")
        if action.shell_command:
            print(f"  shell_command={action.shell_command}")

        result = env.step(action)
        observation = result
        reward = result.reward or 0.0

        transcript.append(
            {"role": "assistant", "content": action.model_dump_json(exclude_none=True)}
        )
        transcript.append(
            {
                "role": "user",
                "content": (
                    f"Reward delta: {reward}\n"
                    f"Observation after command:\n{observation_summary(observation)}"
                ),
            }
        )

    print(f"Final grader score: {observation.grader_score:.2f}")
    print(f"Done: {observation.done}")
    return {
        "task": task_name,
        "score": observation.grader_score,
        "steps": env.state.step_count,
        "solved": observation.grader_score >= 1.0,
        "mode": "openai" if client is not None else "scripted-fallback",
    }


def main() -> list[dict[str, Any]]:
    start = time.time()
    client = create_openai_client()
    model = model_name()

    print("=" * 64)
    print("Git-CI-Gym Baseline Inference")
    print(f"Model mode: {'OpenAI API' if client else 'scripted fallback'}")
    print(f"Model name: {model}")
    print(f"Tasks: {TaskRegistry.list_tasks()}")
    print("=" * 64)

    env = GitCIEnvironment()
    results = []

    try:
        for task_name in TaskRegistry.list_tasks():
            results.append(run_task(env, task_name, client, model))
    finally:
        env.close()

    elapsed = time.time() - start
    average_score = sum(result["score"] for result in results) / len(results)

    print("\n" + "=" * 64)
    print("FINAL RESULTS")
    print("=" * 64)
    for result in results:
        status = "SOLVED" if result["solved"] else "FAILED"
        print(
            f"{result['task']:<8} score={result['score']:.2f} "
            f"steps={result['steps']:<2} status={status:<6} mode={result['mode']}"
        )
    print(f"Average score: {average_score:.2f}")
    print(f"Elapsed: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    main()
