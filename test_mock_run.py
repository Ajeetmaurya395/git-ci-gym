import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from server.git_ci_environment import GitCIEnvironment
from server.tasks import TaskRegistry
from git_ci_gym.models import RepairCommand, RepoRepairAction

def main():
    print("=== Git-CI-Gym Mock Solve Verification ===")
    env = GitCIEnvironment()
    
    # 1. Reset Easy Task
    print("\n[Step 1] Resetting Easy Task...")
    obs = env.reset(task="easy")
    print(f"Conflicts: {obs.conflict_files}")
    print(f"Stage: {obs.stage}")
    print(f"CI passing: {obs.ci_passing}")
    
    # 2. Get solve content from registry
    scenario = TaskRegistry.get('easy')
    solve_content = scenario.solution_main_py
    
    # 3. Simulate Agent 'edit_file' Tool Call
    print("\n[Step 2] Agent calling 'edit_file' to resolve conflict...")
    action = RepoRepairAction(
        command=RepairCommand.write_file,
        path="app/main.py",
        content=solve_content,
    )
    obs = env.step(action)
    print("Action complete.")
    
    # 4. Verify Final State
    print("\n[Step 3] Verifying Final Environment State...")
    print(f"Merge resolved: {obs.merge_resolved}")
    print(f"CI passing: {obs.ci_passing}")
    print(f"Final Reward: {obs.reward}")
    print(f"Absolute Grader Score: {obs.grader_score}")
    print(f"Reward Breakdown: {obs.grader_breakdown}")
    print(f"Done: {obs.done}")
    
    if obs.grader_score >= 1.0:
        print("\n✅ SUCCESS: Mock Run Completed perfectly!")
    else:
        print("\n❌ FAILURE: Mock Run failed to reach 1.0 reward.")

if __name__ == "__main__":
    main()
