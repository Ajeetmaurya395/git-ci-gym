"""
FastAPI application for the typed Git-CI-Gym repo repair environment.
"""

from __future__ import annotations

from textwrap import dedent
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic import ValidationError

try:
    from fastapi import HTTPException
    from fastapi.responses import HTMLResponse
    from openenv.core.env_server.http_server import create_app

    from ..models import RepoRepairAction, RepoRepairObservation, SourceKind
    from .git_ci_environment import GitCIEnvironment
except ImportError:
    from fastapi import HTTPException
    from fastapi.responses import HTMLResponse
    from openenv.core.env_server.http_server import create_app

    from models import RepoRepairAction, RepoRepairObservation, SourceKind
    from server.git_ci_environment import GitCIEnvironment


app = create_app(
    GitCIEnvironment,
    RepoRepairAction,
    RepoRepairObservation,
    env_name="git_ci_gym",
    max_concurrent_envs=4,
)


class BrowserResetRequest(BaseModel):
    session_id: str | None = None
    task: str = "easy"
    source_kind: SourceKind = SourceKind.builtin
    source_ref: str | None = None
    repo_label: str | None = None
    objective: str | None = None


class BrowserStepRequest(BaseModel):
    session_id: str
    action: RepoRepairAction


class BrowserSessionStore:
    """Small in-memory session store for the browser control panel."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def reset(self, request: BrowserResetRequest) -> tuple[str, RepoRepairObservation, dict[str, Any]]:
        with self._lock:
            session_id = request.session_id or str(uuid4())
            if session_id in self._sessions:
                env = self._sessions[session_id]["env"]
            else:
                env = GitCIEnvironment()
                self._sessions[session_id] = {"env": env, "last_observation": None}

        observation = env.reset(
            task=request.task,
            source_kind=request.source_kind.value,
            source_ref=request.source_ref,
            repo_label=request.repo_label,
            objective=request.objective,
        )
        with self._lock:
            self._sessions[session_id]["last_observation"] = observation
        return session_id, observation, env.state.model_dump(mode="json")

    def step(self, request: BrowserStepRequest) -> tuple[RepoRepairObservation, dict[str, Any]]:
        session = self._require_session(request.session_id)
        env = session["env"]
        observation = env.step(request.action)
        with self._lock:
            self._sessions[request.session_id]["last_observation"] = observation
        return observation, env.state.model_dump(mode="json")

    def state(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        env = session["env"]
        observation = session["last_observation"]
        return {
            "session_id": session_id,
            "state": env.state.model_dump(mode="json"),
            "observation": observation.model_dump(mode="json") if observation else None,
        }

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session["env"].close()

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()
        for _, session in sessions:
            session["env"].close()

    def _require_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown session_id. Reset an episode first.")
        return session


browser_sessions = BrowserSessionStore()


@app.on_event("shutdown")
def _close_browser_sessions() -> None:
    browser_sessions.close_all()


@app.post("/ui/reset")
def ui_reset(request: BrowserResetRequest) -> dict[str, Any]:
    session_id, observation, state = browser_sessions.reset(request)
    return {
        "session_id": session_id,
        "observation": observation.model_dump(mode="json"),
        "state": state,
    }


@app.post("/ui/step")
def ui_step(request: dict[str, Any]) -> dict[str, Any]:
    try:
        step_request = BrowserStepRequest(
            session_id=request.get("session_id", ""),
            action=_normalize_browser_action(request),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    observation, state = browser_sessions.step(step_request)
    return {
        "session_id": step_request.session_id,
        "observation": observation.model_dump(mode="json"),
        "state": state,
    }


@app.get("/ui/state")
def ui_state(session_id: str) -> dict[str, Any]:
    return browser_sessions.state(session_id)


@app.delete("/ui/session")
def ui_close_session(session_id: str) -> dict[str, bool]:
    browser_sessions.close(session_id)
    return {"closed": True}


def _normalize_browser_action(request: dict[str, Any]) -> RepoRepairAction:
    raw_action = request.get("action")
    if raw_action is None:
        raw_action = {
            key: value
            for key, value in request.items()
            if key in {"command", "path", "content", "shell_command", "notes"}
        }
    if not isinstance(raw_action, dict):
        raise ValidationError.from_exception_data(
            "RepoRepairAction",
            [
                {
                    "loc": ("action",),
                    "msg": "action payload must be an object",
                    "type": "value_error",
                    "input": raw_action,
                }
            ],
        )
    cleaned_action = {
        key: (value.strip() if isinstance(value, str) else value) for key, value in raw_action.items()
    }
    for optional_key in {"path", "content", "shell_command", "notes"}:
        if cleaned_action.get(optional_key) == "":
            cleaned_action[optional_key] = None
    return RepoRepairAction.model_validate(cleaned_action)


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Render a project-focused operational homepage."""
    html = dedent(
        """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Git-CI-Gym | Repo Repair Loop</title>
          <style>
            :root {
              --bg: #f5f1e7;
              --ink: #182126;
              --muted: #5d6b71;
              --panel: rgba(255, 255, 255, 0.84);
              --line: rgba(24, 33, 38, 0.12);
              --accent: #0f766e;
              --accent-soft: rgba(15, 118, 110, 0.10);
              --warm: #b45309;
              --danger: #b42318;
              --shadow: 0 24px 70px rgba(24, 33, 38, 0.10);
            }
            * { box-sizing: border-box; }
            body {
              margin: 0;
              color: var(--ink);
              font-family: Georgia, "Iowan Old Style", serif;
              background:
                radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 28rem),
                radial-gradient(circle at bottom right, rgba(180,83,9,0.16), transparent 22rem),
                linear-gradient(180deg, #fbf9f3 0%, var(--bg) 100%);
            }
            main {
              max-width: 1180px;
              margin: 0 auto;
              padding: 36px 24px 72px;
            }
            .hero, .panel {
              background: var(--panel);
              border: 1px solid var(--line);
              border-radius: 26px;
              box-shadow: var(--shadow);
            }
            .hero {
              padding: 32px;
              margin-bottom: 22px;
            }
            .eyebrow {
              display: inline-block;
              padding: 8px 12px;
              border-radius: 999px;
              background: var(--accent-soft);
              color: var(--accent);
              font-size: 13px;
              letter-spacing: 0.08em;
              text-transform: uppercase;
            }
            h1, h2, h3 {
              margin: 0;
              font-family: "Helvetica Neue", Arial, sans-serif;
              letter-spacing: -0.04em;
            }
            h1 {
              margin-top: 16px;
              font-size: clamp(42px, 8vw, 82px);
              line-height: 0.95;
            }
            .lead {
              max-width: 52rem;
              margin: 16px 0 0;
              color: var(--muted);
              font-size: 20px;
              line-height: 1.6;
            }
            .stack {
              display: grid;
              gap: 20px;
            }
            .ops {
              display: grid;
              grid-template-columns: 360px 1fr;
              gap: 20px;
            }
            .panel {
              padding: 22px;
            }
            label {
              display: block;
              margin-bottom: 8px;
              font-family: "Helvetica Neue", Arial, sans-serif;
              font-weight: 700;
              font-size: 14px;
            }
            input, select, textarea, button {
              width: 100%;
              font: inherit;
            }
            input, select, textarea {
              padding: 12px 14px;
              border-radius: 14px;
              border: 1px solid var(--line);
              background: rgba(255,255,255,0.72);
              color: var(--ink);
            }
            textarea {
              min-height: 132px;
              resize: vertical;
            }
            button {
              border: 0;
              border-radius: 14px;
              padding: 12px 16px;
              font-family: "Helvetica Neue", Arial, sans-serif;
              font-weight: 700;
              cursor: pointer;
            }
            .primary {
              background: var(--accent);
              color: #fff;
            }
            .secondary {
              background: rgba(24, 33, 38, 0.08);
              color: var(--ink);
            }
            .danger {
              background: rgba(180, 35, 24, 0.12);
              color: var(--danger);
            }
            .grid2, .grid3 {
              display: grid;
              gap: 14px;
            }
            .grid2 { grid-template-columns: 1fr 1fr; }
            .grid3 { grid-template-columns: repeat(3, 1fr); }
            .muted {
              color: var(--muted);
              line-height: 1.6;
            }
            .mono, pre {
              font-family: "SFMono-Regular", Menlo, monospace;
            }
            pre {
              margin: 0;
              padding: 16px;
              border-radius: 16px;
              background: #182126;
              color: #f7f7f7;
              overflow-x: auto;
              min-height: 120px;
              white-space: pre-wrap;
            }
            #terminalLog {
              max-height: 400px;
              overflow-y: auto;
              color: #10b981;
            }
            .chips {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
              margin-top: 14px;
            }
            .chip {
              padding: 8px 12px;
              border-radius: 999px;
              background: rgba(24,33,38,0.06);
              font-family: "Helvetica Neue", Arial, sans-serif;
              font-size: 13px;
            }
            .kpi {
              display: grid;
              grid-template-columns: repeat(4, 1fr);
              gap: 14px;
            }
            .kpi .panel {
              padding: 18px;
            }
            .value {
              margin-top: 8px;
              font-family: "Helvetica Neue", Arial, sans-serif;
              font-size: 30px;
              font-weight: 800;
            }
            .hidden { display: none; }
            .small {
              font-size: 13px;
            }
            @media (max-width: 980px) {
              .ops, .kpi, .grid2, .grid3 {
                grid-template-columns: 1fr;
              }
            }
          </style>
        </head>
        <body>
          <main>
            <section class="hero">
              <span class="eyebrow">Repo Repair Environment</span>
              <h1>Give the app a failing repo and run the repair loop here.</h1>
              <p class="lead">
                Start a browser session with a curriculum task, a local repo path on this machine,
                or a remote Git URL. Then issue repair actions, inspect repo state, and watch the
                grader score move toward 1.0.
              </p>
              <div class="chips">
                <a class="chip" href="/docs">Swagger UI</a>
                <a class="chip" href="/redoc">API Reference</a>
                <a class="chip" href="/openapi.json">OpenAPI JSON</a>
              </div>
            </section>

            <section class="ops">
              <div class="stack">
                <article class="panel">
                  <h2>1. Start Episode</h2>
                  <p class="muted small">
                    `repo_path` must exist on the same machine running this server.
                    `repo_url` will clone a remote repository into a temporary workspace.
                  </p>
                  <div class="stack">
                    <div>
                      <label for="sourceKind">Source Kind</label>
                      <select id="sourceKind">
                        <option value="builtin">builtin</option>
                        <option value="repo_path">repo_path</option>
                        <option value="repo_url">repo_url</option>
                      </select>
                    </div>
                    <div id="taskWrap">
                      <label for="task">Curriculum Task</label>
                      <select id="task">
                        <option value="easy">easy</option>
                        <option value="medium">medium</option>
                        <option value="hard">hard</option>
                      </select>
                    </div>
                    <div id="sourceRefWrap" class="hidden">
                      <label for="sourceRef">Repo Path or Repo URL</label>
                      <input id="sourceRef" placeholder="/absolute/path/to/repo or https://github.com/org/repo.git" />
                      <p class="muted small" id="sourceHint">
                        Paste a local absolute path, a repo URL, or a GitHub pull request URL.
                        URL input is automatically treated as <span class="mono">repo_url</span>.
                      </p>
                    </div>
                    <div>
                      <label for="repoLabel">Label</label>
                      <input id="repoLabel" placeholder="Optional human-readable label" />
                    </div>
                    <div>
                      <label for="objective">Objective</label>
                      <textarea id="objective" placeholder="Optional custom objective. Leave blank to use the default task objective."></textarea>
                    </div>
                    <div class="grid2">
                      <button type="button" id="resetBtn" class="primary">Start / Reset Session</button>
                      <button type="button" id="closeBtn" class="danger">Close Session</button>
                    </div>
                  </div>
                </article>

                <article class="panel">
                  <h2>2. Run Action</h2>
                  <div class="stack">
                    <div>
                      <label for="command">Command</label>
                      <select id="command">
                        <option value="list_files">list_files</option>
                        <option value="read_file">read_file</option>
                        <option value="write_file">write_file</option>
                        <option value="run_command">run_command</option>
                        <option value="get_status">get_status</option>
                        <option value="submit">submit</option>
                      </select>
                    </div>
                    <div id="pathWrap">
                      <label for="path">Path</label>
                      <input id="path" placeholder="app/main.py" />
                    </div>
                    <div id="shellWrap" class="hidden">
                      <label for="shellCommand">Shell Command</label>
                      <input id="shellCommand" placeholder="pytest -q --tb=no" />
                    </div>
                    <div id="contentWrap" class="hidden">
                      <label for="content">Content</label>
                      <textarea id="content" placeholder="Full file contents for write_file"></textarea>
                    </div>
                    <div>
                      <label for="notes">Notes</label>
                      <input id="notes" placeholder="Optional rationale for this action" />
                    </div>
                    <div class="grid3">
                      <button type="button" class="secondary quick" data-command="list_files">Quick: list</button>
                      <button type="button" class="secondary quick" data-command="get_status">Quick: status</button>
                      <button type="button" class="secondary quick" data-command="submit">Quick: submit</button>
                    </div>
                    <button type="button" id="stepBtn" class="primary">Run Action</button>
                  </div>
                </article>
              </div>

              <div class="stack">
                <div class="kpi">
                  <article class="panel">
                    <h3>Session</h3>
                    <div class="value" id="sessionId">none</div>
                  </article>
                  <article class="panel">
                    <h3>Stage</h3>
                    <div class="value" id="stage">intake</div>
                  </article>
                  <article class="panel">
                    <h3>Score</h3>
                    <div class="value" id="score">0.00</div>
                  </article>
                  <article class="panel">
                    <h3>CI</h3>
                    <div class="value" id="ciStatus">red</div>
                  </article>
                </div>

                <article class="panel">
                  <h2>Observation</h2>
                  <div class="grid2">
                    <div>
                      <label>Repo</label>
                      <div id="repoLabelView" class="muted">not started</div>
                    </div>
                    <div>
                      <label>Objective</label>
                      <div id="objectiveView" class="muted">start a session to load a repo</div>
                    </div>
                  </div>
                  <div class="grid2" style="margin-top: 14px;">
                    <div>
                      <label>Conflict Files</label>
                      <pre id="conflicts">[]</pre>
                    </div>
                    <div>
                      <label>Failing Tests</label>
                      <pre id="failingTests">[]</pre>
                    </div>
                  </div>
                </article>

                <article class="panel">
                  <h2>Terminal Log</h2>
                  <pre id="terminalLog">No action has run yet.</pre>
                </article>

                <article class="panel">
                  <h2>State Snapshot</h2>
                  <pre id="stateDump">{}</pre>
                </article>
              </div>
            </section>
          </main>

          <script>
            const sourceKind = document.getElementById("sourceKind");
            const taskWrap = document.getElementById("taskWrap");
            const sourceRefWrap = document.getElementById("sourceRefWrap");
            const task = document.getElementById("task");
            const sourceRef = document.getElementById("sourceRef");
            const repoLabel = document.getElementById("repoLabel");
            const objective = document.getElementById("objective");
            const resetBtn = document.getElementById("resetBtn");
            const closeBtn = document.getElementById("closeBtn");
            const command = document.getElementById("command");
            const pathWrap = document.getElementById("pathWrap");
            const shellWrap = document.getElementById("shellWrap");
            const contentWrap = document.getElementById("contentWrap");
            const pathField = document.getElementById("path");
            const shellCommand = document.getElementById("shellCommand");
            const content = document.getElementById("content");
            const notes = document.getElementById("notes");
            const stepBtn = document.getElementById("stepBtn");
            const sessionId = document.getElementById("sessionId");
            const stage = document.getElementById("stage");
            const score = document.getElementById("score");
            const ciStatus = document.getElementById("ciStatus");
            const repoLabelView = document.getElementById("repoLabelView");
            const objectiveView = document.getElementById("objectiveView");
            const conflicts = document.getElementById("conflicts");
            const failingTests = document.getElementById("failingTests");
            const terminalLog = document.getElementById("terminalLog");
            const stateDump = document.getElementById("stateDump");

            let currentSessionId = localStorage.getItem("git_ci_gym_session_id") || "";

            function renderSourceInputs() {
              const builtin = sourceKind.value === "builtin";
              taskWrap.classList.toggle("hidden", !builtin);
              sourceRefWrap.classList.toggle("hidden", builtin);
            }

            function normalizeSourceSelection() {
              const ref = (sourceRef.value || "").trim();
              const loweredRef = ref.toLowerCase();
              const looksLikeUrl = loweredRef.startsWith("http://") || loweredRef.startsWith("https://");
              if (looksLikeUrl && sourceKind.value === "repo_path") {
                sourceKind.value = "repo_url";
              }
            }

            function renderCommandInputs() {
              pathWrap.classList.toggle("hidden", !["read_file", "write_file"].includes(command.value));
              shellWrap.classList.toggle("hidden", command.value !== "run_command");
              contentWrap.classList.toggle("hidden", command.value !== "write_file");
            }
            
            function appendToLog(text) {
              const current = terminalLog.textContent;
              if (current === "No action has run yet." || current === "Session closed.") {
                terminalLog.textContent = text;
              } else {
                terminalLog.textContent = current + "\\n" + text;
              }
              terminalLog.scrollTop = terminalLog.scrollHeight;
            }

            function applyObservation(observation, state) {
              if (!observation) return;
              sessionId.textContent = currentSessionId || "none";
              stage.textContent = observation.stage;
              score.textContent = Number(observation.grader_score || 0).toFixed(2);
              ciStatus.textContent = observation.ci_passing ? "green" : "red";
              repoLabelView.textContent = observation.repo_label || "unlabeled";
              objectiveView.textContent = observation.objective || "";
              conflicts.textContent = JSON.stringify(observation.conflict_files || [], null, 2);
              failingTests.textContent = JSON.stringify(observation.failing_tests || [], null, 2);
              
              if (observation.last_command === "reset") {
                appendToLog(`> reset\\nEnvironment initialized. Stage: ${observation.stage}\\n`);
              } else if (observation.last_command) {
                let logEntry = `> ${observation.last_command}`;
                if (observation.last_result) logEntry += `\\n${observation.last_result}`;
                if (observation.notes) logEntry += `\\n[Notes] ${observation.notes}`;
                appendToLog(logEntry + "\\n");
              }
              
              stateDump.textContent = JSON.stringify(state || {}, null, 2);
            }

            async function jsonFetch(url, options) {
              const response = await fetch(url, {
                headers: {"Content-Type": "application/json"},
                ...options,
              });
              const payload = await response.json();
              if (!response.ok) {
                throw new Error(payload.detail || JSON.stringify(payload));
              }
              return payload;
            }

            async function startSession() {
              normalizeSourceSelection();
              const payload = {
                session_id: currentSessionId || null,
                task: task.value,
                source_kind: sourceKind.value,
                source_ref: sourceRef.value || null,
                repo_label: repoLabel.value || null,
                objective: objective.value || null,
              };
              const result = await jsonFetch("/ui/reset", {
                method: "POST",
                body: JSON.stringify(payload),
              });
              currentSessionId = result.session_id;
              localStorage.setItem("git_ci_gym_session_id", currentSessionId);
              applyObservation(result.observation, result.state);
            }

            function buildActionPayload() {
              const action = {
                command: command.value,
              };
              if (["read_file", "write_file"].includes(command.value) && pathField.value) {
                action.path = pathField.value;
              }
              if (command.value === "write_file") {
                action.content = content.value;
              }
              if (command.value === "run_command") {
                action.shell_command = shellCommand.value;
              }
              if (notes.value) {
                action.notes = notes.value;
              }
              return action;
            }

            async function runAction() {
              if (!currentSessionId) {
                throw new Error("Start a session first.");
              }
              const result = await jsonFetch("/ui/step", {
                method: "POST",
                body: JSON.stringify({
                  session_id: currentSessionId,
                  action: buildActionPayload(),
                }),
              });
              applyObservation(result.observation, result.state);
            }

            async function closeSession() {
              if (!currentSessionId) return;
              await fetch(`/ui/session?session_id=${encodeURIComponent(currentSessionId)}`, {method: "DELETE"});
              currentSessionId = "";
              localStorage.removeItem("git_ci_gym_session_id");
              sessionId.textContent = "none";
              stage.textContent = "intake";
              score.textContent = "0.00";
              ciStatus.textContent = "red";
              repoLabelView.textContent = "not started";
              objectiveView.textContent = "start a session to load a repo";
              conflicts.textContent = "[]";
              failingTests.textContent = "[]";
              terminalLog.textContent = "Session closed.";
              stateDump.textContent = "{}";
            }

            async function loadExistingState() {
              if (!currentSessionId) return;
              try {
                const result = await fetch(`/ui/state?session_id=${encodeURIComponent(currentSessionId)}`);
                if (!result.ok) {
                  currentSessionId = "";
                  localStorage.removeItem("git_ci_gym_session_id");
                  return;
                }
                const payload = await result.json();
                applyObservation(payload.observation, payload.state);
              } catch (error) {
                console.error(error);
              }
            }

            sourceKind.addEventListener("change", renderSourceInputs);
            sourceRef.addEventListener("blur", normalizeSourceSelection);
            command.addEventListener("change", renderCommandInputs);
            resetBtn.addEventListener("click", async () => {
              try {
                await startSession();
              } catch (error) {
                appendToLog(`[Error] ${error.message}\\n`);
              }
            });
            closeBtn.addEventListener("click", async () => {
              try {
                await closeSession();
              } catch (error) {
                appendToLog(`[Error] ${error.message}\\n`);
              }
            });
            stepBtn.addEventListener("click", async () => {
              try {
                await runAction();
              } catch (error) {
                appendToLog(`[Error] ${error.message}\\n`);
              }
            });
            
            document.querySelectorAll(".quick").forEach((button) => {
              button.addEventListener("click", async () => {
                command.value = button.dataset.command;
                renderCommandInputs();
                try {
                  await runAction();
                } catch (error) {
                  appendToLog(`[Error] ${error.message}\\n`);
                }
              });
            });

            renderSourceInputs();
            renderCommandInputs();
            loadExistingState();
          </script>
        </body>
        </html>
        """
    ).strip()
    return HTMLResponse(content=html)


def main():
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
