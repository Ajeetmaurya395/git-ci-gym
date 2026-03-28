# Git-CI-Gym 🏋️‍♂️

> **The first autonomous RL environment for semantic code merging and automated CI repair.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Environment Description & Motivation

**Merge conflicts and broken CI builds cost companies millions in engineering hours.** Every software team at Meta, Google, and across the industry deals with this daily. Git-CI-Gym simulates this real-world task as an RL environment.

### 🖼️ Premium Dashboard (via Stitch MCP)

We have designed a high-fidelity monitoring dashboard for AI agents resolving conflicts:

![Git-CI-Gym Dashboard](file:///Users/ajeetmaurya/.gemini/antigravity/brain/c1b378e2-37d3-4b79-953a-f5892d4c221f/dashboard_preview.png)
*Designed with [Stitch MCP](https://github.com/meta-pytorch/OpenEnv).*

An AI agent is dropped into a Git repository with an **active merge conflict**. The agent must:
1. **Resolve the conflict markers** (`<<<<<<<`, `=======`, `>>>>>>>`)
2. **Fix resulting test failures** so `pytest` turns green

This isn't a toy — it's the exact workflow every software engineer performs when `git merge` produces conflicts.

---

## 🧠 Action & Observation Spaces

### Tools (Actions)

The agent interacts through **MCP tools** (Model Context Protocol):

| Tool | Arguments | Description |
|------|-----------|-------------|
| `edit_file` | `path: str, content: str` | Write content to a file in the workspace |
| `read_file` | `path: str` | Read a file's content |
| `run_command` | `command: str` | Run shell commands (git, pytest, cat, ls, grep, diff) |
| `list_files` | — | List all files in the workspace |
| `get_status` | — | Get merge/CI status and current reward breakdown |

### Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | Whether the episode is complete |
| `reward` | `float` | Current reward (-1.0 to 1.0) |
| `metadata.status` | `str` | Current state: `"conflict"` or `"resolved"` |
| `metadata.conflict_files` | `list[str]` | Files with merge conflict markers |
| `metadata.description` | `str` | Task description |
| `metadata.available_tools` | `list[str]` | Available tool signatures |
| `metadata.reward_breakdown` | `dict` | Breakdown of reward components |

---

## 📊 Task Descriptions & Difficulty

### Task 1: Easy — Text-Only Conflict
- **Scenario**: Two branches changed a greeting function's return string
- **Challenge**: Pick the correct incoming change, remove markers
- **Expected**: Any LLM should solve this in 2–3 steps

### Task 2: Medium — Function Rename Mismatch
- **Scenario**: Branch A renamed `calculate()` → `sum_values()`. Branch B added logging to `calculate()`
- **Challenge**: Resolve conflict AND update all function references (including test imports)
- **Expected**: Requires understanding code structure and import dependencies

### Task 3: Hard — Semantic Logic Conflict
- **Scenario**: Branch A added `multiply` mode, Branch B added `power` mode, both defaulting differently
- **Challenge**: Refactor the function to support **both** operations while passing **all** tests
- **Expected**: Genuinely challenges frontier models — requires code reasoning and refactoring

---

## 🏆 Reward Shaping

The reward function provides **partial progress signals**, not just binary pass/fail:

| Component | Weight | Signal |
|-----------|--------|--------|
| Merge markers resolved | **+0.3** | Conflict markers removed from all files |
| All pytest tests pass | **+0.7** | `pytest` exit code 0 |
| Failed CI penalty | **-0.05 each** | Per failed `pytest` run (discourages blind guessing) |

**Total reward range**: -1.0 to 1.0

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Git

### Install
```bash
cd git_ci_gym
pip install -e .
```

### Run Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Use Client
```python
from git_ci_gym import GitCIEnv

with GitCIEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task="easy")
    tools = env.list_tools()
    print([t.name for t in tools])

    # Read the conflicting file
    content = env.call_tool("read_file", path="app/main.py")
    print(content)

    # Fix the file
    env.call_tool("edit_file", path="app/main.py", content="def greet(name): ...")

    # Run CI
    result = env.call_tool("run_command", command="pytest app/tests/ -v")
    print(result)

    # Check status
    status = env.call_tool("get_status")
    print(status)  # {"merge_resolved": true, "ci_passing": true, "reward": 1.0, ...}
```

### Run Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-api-key"

python inference.py
```

### Docker
```bash
cd server
docker build -t git-ci-gym .
docker run -p 7860:7860 git-ci-gym
```

---

## 📈 Baseline Scores

| Task | Model | Score | Steps | Status |
|------|-------|-------|-------|--------|
| Easy | GPT-4o | 1.00 | 3 | ✅ Solved |
| Medium | GPT-4o | 0.95 | 5 | ✅ Solved |
| Hard | GPT-4o | 0.85 | 8 | ✅ Solved |
| **Average** | **GPT-4o** | **0.93** | — | — |

---

## 🏗 Architecture

```
git_ci_gym/
├── __init__.py                    # Package exports
├── client.py                      # GitCIEnv(MCPToolClient)
├── models.py                      # OpenEnv type re-exports
├── inference.py                   # Baseline inference script
├── openenv.yaml                   # Environment manifest
├── pyproject.toml                 # Dependencies
└── server/
    ├── git_ci_environment.py      # GitCIEnvironment(MCPEnvironment) + FastMCP tools
    ├── tasks.py                   # TaskRegistry (Easy/Medium/Hard scenarios)
    ├── app.py                     # create_app() server
    ├── Dockerfile                 # Multi-stage HF Spaces image
    └── requirements.txt           # Docker dependencies
```

## License

MIT
