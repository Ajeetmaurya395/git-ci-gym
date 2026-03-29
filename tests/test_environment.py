from git_ci_gym.models import RepairCommand, RepoRepairAction, RepairStage, SourceKind
from server.git_ci_environment import GitCIEnvironment
from server.tasks import TaskRegistry


def test_reset_reports_conflict_state():
    env = GitCIEnvironment()

    try:
        for task_name in TaskRegistry.list_tasks():
            obs = env.reset(task=task_name)

            assert obs.done is False
            assert obs.reward == 0.0
            assert obs.task_id == task_name
            assert obs.stage == RepairStage.repair
            assert obs.conflict_files == ["app/main.py"]
            assert obs.merge_resolved is False
            assert obs.ci_passing is False
    finally:
        env.close()


def test_tool_calls_return_progress_reward():
    env = GitCIEnvironment()

    try:
        env.reset(task="easy")
        obs = env.step(RepoRepairAction(command=RepairCommand.read_file, path="app/main.py"))

        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.stage == RepairStage.repair
        assert obs.last_command == "read_file"
        assert "def greet" in obs.last_result
    finally:
        env.close()


def test_solution_edit_solves_each_task():
    env = GitCIEnvironment()

    try:
        for task_name in TaskRegistry.list_tasks():
            env.reset(task=task_name)
            scenario = TaskRegistry.get(task_name)

            obs = env.step(
                RepoRepairAction(
                    command=RepairCommand.write_file,
                    path="app/main.py",
                    content=scenario.solution_main_py,
                )
            )

            assert obs.done is True
            assert obs.task_id == task_name
            assert obs.merge_resolved is True
            assert obs.ci_passing is True
            assert obs.grader_score == 1.0
    finally:
        env.close()


def test_repo_path_url_is_coerced_to_repo_url():
    env = GitCIEnvironment()

    try:
        source_kind, source_ref = env._normalize_source_input(
            SourceKind.repo_path,
            "https://github.com/pocketpaw/pocketpaw/pull/762#pullrequestreview-4025252046",
        )

        assert source_kind.value == "repo_url"
        assert source_ref == "https://github.com/pocketpaw/pocketpaw/pull/762#pullrequestreview-4025252046"
    finally:
        env.close()


def test_github_pull_request_urls_are_normalized_for_clone():
    env = GitCIEnvironment()

    try:
        remote_spec = env._build_remote_spec(
            "https://github.com/pocketpaw/pocketpaw/pull/762#pullrequestreview-4025252046"
        )

        assert remote_spec == {
            "clone_url": "https://github.com/pocketpaw/pocketpaw.git",
            "pull_request": "762",
        }
    finally:
        env.close()


def test_multi_step_trajectory():
    env = GitCIEnvironment()

    try:
        env.reset(task="easy")
        
        # Step 1: Read the file
        obs = env.step(RepoRepairAction(command=RepairCommand.read_file, path="app/main.py"))
        assert "def greet" in obs.last_result
        assert obs.reward == 0.0

        # Step 2: Write a wrong solution
        obs = env.step(RepoRepairAction(
            command=RepairCommand.write_file, 
            path="app/main.py", 
            content="def greet(name: str):\n    return f'Hello, {name}'\n"
        ))
        assert obs.done is False
        
        # Step 3: Shell command
        obs = env.step(RepoRepairAction(command=RepairCommand.run_command, shell_command="pytest"))
        assert obs.stage in [RepairStage.verify, RepairStage.repair]

        # Step 4: Correct solution
        scenario = TaskRegistry.get("easy")
        obs = env.step(RepoRepairAction(
            command=RepairCommand.write_file, 
            path="app/main.py", 
            content=scenario.solution_main_py
        ))
        assert obs.ci_passing is True
        assert obs.merge_resolved is True
        assert obs.grader_score > 0.9

    finally:
        env.close()


def test_ingest_repo_url_mocked(monkeypatch):
    import os
    env = GitCIEnvironment()
    
    calls = []
    original_run_internal = env._run_internal
    
    def mock_run_internal(argv, cwd=None, check=True, timeout=30):
        if argv[0] == "git" and argv[1] == "clone":
            calls.append(argv)
            os.makedirs(os.path.join(env._workspace, ".git"), exist_ok=True)
            with open(os.path.join(env._workspace, "app.py"), "w") as f:
                f.write("print('hello')")
            return 0, ""
        return original_run_internal(argv, cwd, check, timeout)
        
    monkeypatch.setattr(env, "_run_internal", mock_run_internal)
    
    try:
        obs = env.reset(source_kind=SourceKind.repo_url, source_ref="https://github.com/test/repo.git")
        assert len(calls) == 1
        assert "clone" in calls[0]
        assert obs.source_kind == SourceKind.repo_url
        assert obs.source_ref == "https://github.com/test/repo.git"
        assert "app.py" in obs.workspace_snapshot
    finally:
        env.close()
