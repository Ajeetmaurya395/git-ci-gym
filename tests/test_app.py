from pathlib import Path

from fastapi.testclient import TestClient

from server.app import app
from server.git_ci_environment import GitCIEnvironment


def test_root_redirects_to_docs():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Git-CI-Gym" in response.text
    assert "Repo Repair Environment" in response.text
    assert "Swagger UI" in response.text
    assert "Start / Reset Session" in response.text
    assert "/docs" in response.text


def test_reset_endpoint_returns_initial_observation():
    client = TestClient(app)

    response = client.post("/reset", json={"task": "easy"})
    body = response.json()

    assert response.status_code == 200
    assert body["done"] is False
    assert body["reward"] == 0.0
    assert body["observation"]["task_id"] == "easy"
    assert body["observation"]["stage"] == "repair"
    assert body["observation"]["conflict_files"] == ["app/main.py"]


def test_browser_ui_session_flow():
    client = TestClient(app)

    reset_response = client.post(
        "/ui/reset",
        json={"task": "easy", "source_kind": "builtin"},
    )
    reset_body = reset_response.json()
    session_id = reset_body["session_id"]

    assert reset_response.status_code == 200
    assert reset_body["observation"]["task_id"] == "easy"
    assert reset_body["state"]["task_id"] == "easy"

    step_response = client.post(
        "/ui/step",
        json={
            "session_id": session_id,
            "action": {"command": "get_status"},
        },
    )
    step_body = step_response.json()

    assert step_response.status_code == 200
    assert step_body["observation"]["last_command"] == "get_status"
    assert step_body["state"]["step_count"] == 1

    state_response = client.get(f"/ui/state?session_id={session_id}")
    state_body = state_response.json()

    assert state_response.status_code == 200
    assert state_body["state"]["task_id"] == "easy"
    assert state_body["observation"]["task_id"] == "easy"

    close_response = client.delete(f"/ui/session?session_id={session_id}")
    assert close_response.status_code == 200


def test_browser_ui_accepts_github_url_even_if_repo_path_is_selected(monkeypatch):
    client = TestClient(app)

    def fake_ingest_repo_url(self, source_ref):
        Path(self._workspace).mkdir(parents=True, exist_ok=True)
        Path(self._workspace, "README.md").write_text(f"cloned from {source_ref}", encoding="utf-8")

    def fake_run_pytest(self):
        return True, 0, [], "all checks passed"

    monkeypatch.setattr(GitCIEnvironment, "_ingest_repo_url", fake_ingest_repo_url)
    monkeypatch.setattr(GitCIEnvironment, "_run_pytest", fake_run_pytest)

    response = client.post(
        "/ui/reset",
        json={
            "task": "easy",
            "source_kind": "repo_path",
            "source_ref": "https://github.com/pocketpaw/pocketpaw/pull/762#pullrequestreview-4025252046",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["state"]["source_kind"] == "repo_url"
    assert (
        body["state"]["source_ref"]
        == "https://github.com/pocketpaw/pocketpaw/pull/762#pullrequestreview-4025252046"
    )
