"""
Git-CI-Gym Inference Script.

Official baseline script for the Meta PyTorch OpenEnv Hackathon.
Uses the OpenAI Client with environment variables:
  - API_BASE_URL: The API endpoint for the LLM.
  - MODEL_NAME: The model identifier to use for inference.
  - HF_TOKEN: Your Hugging Face / API key.

Runs all 3 tasks (easy, medium, hard) and prints reproducible scores.
Must complete in under 20 minutes on vcpu=2, memory=8gb.
"""

import json
import os
import sys
import time

from openai import OpenAI

# ── Environment Setup ────────────────────────────────────────────────
# Add project root to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from server.git_ci_environment import GitCIEnvironment
from server.tasks import TaskRegistry


def create_client() -> OpenAI:
    """Create an OpenAI client using hackathon environment variables."""
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

    if not hf_token:
        print("⚠️  Warning: No API key found. Set HF_TOKEN or OPENAI_API_KEY.")

    return OpenAI(
        base_url=api_base_url,
        api_key=hf_token,
    )


def get_model_name() -> str:
    """Get the model name from environment variables."""
    return os.environ.get("MODEL_NAME", "gpt-4o")


SYSTEM_PROMPT = """You are a Senior DevOps Engineer who specializes in Git operations and CI/CD pipelines.

You are interacting with a Git repository that has an active merge conflict.
Your job is to:
1. Read the conflicting files to understand the conflict
2. Edit the files to resolve the conflict (remove markers, merge logic correctly)
3. Make sure all pytest tests pass

You have access to these tools:
- read_file(path): Read a file's content
- edit_file(path, content): Write new content to a file
- run_command(command): Run shell commands (git, pytest, cat, ls, grep, diff, etc.)
- list_files(): List all files in the workspace
- get_status(): Check current merge/CI status and reward

IMPORTANT RULES:
- Always read the conflicting file FIRST before editing
- After editing, run pytest to verify your fix
- Check get_status() to see your current score
- Resolve the conflict correctly — don't just delete one side blindly
- The tests define what the correct behavior should be — read them!

Respond with a JSON object: {"tool": "tool_name", "args": {"arg1": "value1"}}
"""


def run_task(env: GitCIEnvironment, client: OpenAI, model: str, task_level: str, max_steps: int = 10) -> dict:
    """
    Run a single task and return the result.

    Returns:
        Dict with task, score, steps, and status.
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task_level.upper()}")
    print(f"{'='*60}")

    # Reset the environment for this task
    obs = env.reset(task=task_level)
    print(f"📋 {obs.metadata.get('description', '')}")
    print(f"🔥 Conflict files: {obs.metadata.get('conflict_files', [])}")

    # Build initial context for the LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Task: {task_level.upper()} - {obs.metadata.get('description', '')}\n"
            f"Conflict files: {obs.metadata.get('conflict_files', [])}\n"
            f"Available tools: {obs.metadata.get('available_tools', [])}\n\n"
            "Start by reading the conflicting file and the test file to understand what's expected."
        )},
    ]

    final_score = 0.0
    steps = 0

    for step in range(max_steps):
        steps = step + 1
        print(f"\n--- Step {steps}/{max_steps} ---")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
            )

            assistant_msg = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_msg})

            # Parse the tool call from the response
            try:
                # Try to extract JSON from the response
                json_start = assistant_msg.find("{")
                json_end = assistant_msg.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found")

                action_data = json.loads(assistant_msg[json_start:json_end])
                tool_name = action_data.get("tool", "")
                tool_args = action_data.get("args", {})

                print(f"🤖 Action: {tool_name}({tool_args})")

                # Execute the tool via the environment's MCP tools
                # We call the tool functions directly since we're running locally
                if tool_name == "read_file":
                    result = _call_tool(env, "read_file", **tool_args)
                elif tool_name == "edit_file":
                    result = _call_tool(env, "edit_file", **tool_args)
                elif tool_name == "run_command":
                    result = _call_tool(env, "run_command", **tool_args)
                elif tool_name == "list_files":
                    result = _call_tool(env, "list_files", **tool_args)
                elif tool_name == "get_status":
                    result = _call_tool(env, "get_status", **tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                # Convert result to string for the messages
                result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                print(f"📄 Result: {result_str[:200]}...")

                messages.append({"role": "user", "content": f"Tool result:\n{result_str}\n\nContinue resolving the conflict. If you think you're done, call get_status() to check your score."})

                # Check if the task is solved
                if tool_name == "get_status" and isinstance(result, dict):
                    final_score = result.get("reward", 0.0)
                    if result.get("merge_resolved") and result.get("ci_passing"):
                        print(f"🎉 SOLVED! Score: {final_score}")
                        break

            except (json.JSONDecodeError, ValueError) as e:
                messages.append({"role": "user", "content": f"Error parsing your response: {e}. Please respond with valid JSON: {{\"tool\": \"tool_name\", \"args\": {{...}}}}"})

        except Exception as e:
            print(f"❌ API Error: {e}")
            break

    # Get final status
    status = _call_tool(env, "get_status")
    if isinstance(status, dict):
        final_score = status.get("reward", final_score)

    print(f"\n📊 Final Score for {task_level}: {final_score}")

    return {
        "task": task_level,
        "score": final_score,
        "steps": steps,
        "solved": final_score >= 0.9,
    }


def _call_tool(env: GitCIEnvironment, tool_name: str, **kwargs):
    """
    Call a tool function directly on the environment's MCP server.
    This is the local-execution path for the inference script.
    """
    # Access the tools registered on the FastMCP instance
    # The tools are closures defined in __init__, so we call them via the mcp server
    mcp = env.mcp_server

    # Use fastmcp's internal tool registry
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            mcp.call_tool(tool_name, kwargs)
        )
        loop.close()
        # Parse the result — fastmcp returns a list of content items
        if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
            texts = []
            for item in result:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif hasattr(item, 'content'):
                    texts.append(str(item.content))
                else:
                    texts.append(str(item))
            combined = "\n".join(texts)
            # Try to parse as JSON
            try:
                return json.loads(combined)
            except (json.JSONDecodeError, TypeError):
                return combined
        return result
    except Exception as e:
        return f"Tool error: {str(e)}"


def main():
    """Run inference on all 3 tasks and report scores."""
    start_time = time.time()

    client = create_client()
    model = get_model_name()

    print("=" * 60)
    print("  Git-CI-Gym — Inference Script")
    print(f"  Model: {model}")
    print(f"  Tasks: {TaskRegistry.list_tasks()}")
    print("=" * 60)

    env = GitCIEnvironment()
    results = []

    for task_level in ["easy", "medium", "hard"]:
        result = run_task(env, client, model, task_level, max_steps=10)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"{'Task':<10} {'Score':<10} {'Steps':<10} {'Status':<10}")
    print("-" * 40)
    for r in results:
        status = "✅ SOLVED" if r["solved"] else "❌ FAILED"
        print(f"{r['task']:<10} {r['score']:<10.2f} {r['steps']:<10} {status:<10}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n📊 Average Score: {avg_score:.2f}")
    print(f"⏱️  Total Time: {elapsed:.1f}s")

    if elapsed > 1200:  # 20 minutes
        print("⚠️  WARNING: Inference took longer than 20 minutes!")

    return results


if __name__ == "__main__":
    main()
