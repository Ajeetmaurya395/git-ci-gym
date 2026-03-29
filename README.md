# Git-CI-Gym

Git-CI-Gym is a typed OpenEnv environment for a real-world task: repairing broken repositories. An agent is dropped into a failing repo snapshot, loops through inspection and repair actions, and is graded on whether it removes merge conflicts and makes CI pass.

## Why this environment exists

Teams lose time every day to unresolved merge conflicts, drift between branches, and red CI pipelines. This environment models that workflow directly:

- ingest a repo snapshot
- inspect the working tree
- edit code and rerun checks
- receive partial-credit reward as progress improves
- stop when the repo is actually repaired

The built-in curriculum ships with three deterministic tasks, and the environment also accepts `repo_path` and `repo_url` reset inputs for repo-centric evaluation flows.

## OpenEnv API

The environment implements the standard OpenEnv `reset()`, `step()`, and `state()` contract with typed Pydantic models.

### Action space

`RepoRepairAction` supports these commands:

- `list_files`
- `read_file(path)`
- `write_file(path, content)`
- `run_command(shell_command)`
- `get_status`
- `submit`

### Observation space

`RepoRepairObservation` includes:

- `task_id`, `difficulty`, `stage`
- `source_kind`, `source_ref`, `repo_label`, `objective`
- `workspace_snapshot`
- `conflict_files`
- `failing_tests`, `failing_test_count`
- `merge_resolved`, `ci_passing`
- `grader_score`, `grader_breakdown`
- `last_command`, `last_result`
- standard OpenEnv fields: `reward`, `done`, `metadata`

### State

`RepoRepairState` tracks the same repo health signals plus persistent episode fields such as `initial_failing_test_count`, `failed_command_count`, and `workspace_path`.

## Tasks

### Easy

Text-only merge conflict in a greeting function. The agent removes conflict markers and restores the expected output.

### Medium

One branch renamed a function while another added logging. The agent must preserve the rename, keep the logging, and repair the tests.

### Hard

Two branches introduced competing semantic features (`multiply` and `power`). The agent must merge both behaviors into one correct implementation.

## Reward design

The grader emits an absolute score from `0.0` to `1.0`, and each `step()` returns the delta from the previous score as reward.

Current breakdown:

- `0.35` for removing all merge markers
- up to `0.35` for reducing failing tests relative to the episode baseline
- `0.30` bonus when CI fully passes
- up to `-0.20` penalty for failed or invalid commands

This gives the agent dense feedback across the trajectory instead of only terminal success.

## Usage

### Install

```bash
cd git_ci_gym
./venv/bin/pip install --no-deps -e .
```

### Run the server

```bash
./venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Use the typed client

```python
from git_ci_gym import GitCIEnv, RepairCommand, RepoRepairAction

with GitCIEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task="easy", source_kind="builtin")
    print(result.observation.objective)

    result = env.step(
        RepoRepairAction(command=RepairCommand.read_file, path="app/main.py")
    )
    print(result.observation.last_result)
```

### Reset with a local repo

```json
{
  "task": "custom",
  "source_kind": "repo_path",
  "source_ref": "/absolute/path/to/failing-repo"
}
```

### Baseline inference

`inference.py` uses the OpenAI client when API credentials are present and falls back to a deterministic scripted curriculum baseline when they are not.

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`

Run it with:

```bash
./venv/bin/python inference.py
```

## Browser UI

Opening `http://127.0.0.1:7860/` shows a Git-CI-Gym landing page focused on repo repair loops, input modes, curriculum tasks, and quick links to:

- `/docs`
- `/redoc`
- `/openapi.json`

## Docker and Hugging Face Spaces

The repository already includes a `Dockerfile` and remains shaped for container deployment. The environment server entrypoint is `server.app:app`, and the default port is `7860`, which matches Hugging Face Spaces expectations.

## Local verification

These checks currently pass:

```bash
./venv/bin/python -m pytest -q
./venv/bin/python test_mock_run.py
```

## Project layout

```text
git_ci_gym/
├── __init__.py
├── client.py
├── models.py
├── inference.py
├── openenv.yaml
├── server/
│   ├── app.py
│   ├── git_ci_environment.py
│   └── tasks.py
└── tests/
    ├── test_app.py
    └── test_environment.py
```
