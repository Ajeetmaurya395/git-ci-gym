"""
Git-CI-Gym Task Registry.

Defines Easy, Medium, and Hard scenarios for git merge conflict resolution.
Each task provides the file contents for main branch, feature branch, and tests —
enough to create a reproducible git conflict + CI failure scenario.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TaskScenario:
    """A single task scenario that defines a git conflict + CI test."""
    name: str
    difficulty: str
    description: str
    # Content for the initial commit (shared base)
    base_main_py: str
    base_test_py: str
    # Content for the feature branch changes
    feature_main_py: str
    # Content for the main branch conflicting changes
    main_branch_main_py: str
    # Expected solution (for reference / baseline validation)
    solution_main_py: str
    # Updated tests that must pass after resolution
    final_test_py: str


class TaskRegistry:
    """
    Registry of all available task scenarios.
    Tasks are designed so that:
      - Easy: Just remove conflict markers, pick the right side.
      - Medium: Resolve markers AND fix a renamed function reference.
      - Hard: Resolve markers AND refactor logic to support both features.
    """

    _tasks: Dict[str, TaskScenario] = {}

    @classmethod
    def _register_defaults(cls):
        """Register the built-in Easy/Medium/Hard tasks."""

        # ── EASY: Text-only conflict ──────────────────────────────────
        cls._tasks["easy"] = TaskScenario(
            name="easy",
            difficulty="easy",
            description=(
                "Text-only conflict. One file, one function. "
                "Agent picks the correct incoming change and finishes the merge."
            ),
            base_main_py=(
                "def greet(name: str) -> str:\n"
                '    return f"Hello, {name}!"\n'
            ),
            base_test_py=(
                "from app.main import greet\n\n"
                "def test_greet():\n"
                '    assert greet("World") == "Hello, World!"\n'
            ),
            feature_main_py=(
                "def greet(name: str) -> str:\n"
                '    return f"Hi, {name}! Welcome aboard."\n'
            ),
            main_branch_main_py=(
                "def greet(name: str) -> str:\n"
                '    return f"Hey there, {name}!"\n'
            ),
            solution_main_py=(
                "def greet(name: str) -> str:\n"
                '    return f"Hi, {name}! Welcome aboard."\n'
            ),
            final_test_py=(
                "from app.main import greet\n\n"
                "def test_greet():\n"
                '    assert greet("World") == "Hi, World! Welcome aboard."\n'
            ),
        )

        # ── MEDIUM: Variable / function rename mismatch ───────────────
        cls._tasks["medium"] = TaskScenario(
            name="medium",
            difficulty="medium",
            description=(
                "Function was renamed on one branch. The test still uses the old name. "
                "Agent must resolve conflict AND update the function reference globally."
            ),
            base_main_py=(
                "def calculate(a: int, b: int) -> int:\n"
                "    return a + b\n"
            ),
            base_test_py=(
                "from app.main import calculate\n\n"
                "def test_calculate():\n"
                "    assert calculate(2, 3) == 5\n"
            ),
            feature_main_py=(
                "def sum_values(a: int, b: int) -> int:\n"
                '    """Renamed for clarity."""\n'
                "    return a + b\n"
            ),
            main_branch_main_py=(
                "def calculate(a: int, b: int) -> int:\n"
                '    """Added logging."""\n'
                "    print(f'Calculating {a} + {b}')\n"
                "    return a + b\n"
            ),
            solution_main_py=(
                "def sum_values(a: int, b: int) -> int:\n"
                '    """Renamed for clarity."""\n'
                "    print(f'Calculating {a} + {b}')\n"
                "    return a + b\n"
            ),
            final_test_py=(
                "from app.main import sum_values\n\n"
                "def test_sum_values():\n"
                "    assert sum_values(2, 3) == 5\n"
                "    assert sum_values(-1, 1) == 0\n"
            ),
        )

        # ── HARD: Semantic logic conflict ─────────────────────────────
        cls._tasks["hard"] = TaskScenario(
            name="hard",
            difficulty="hard",
            description=(
                "Two features added independently that work alone but break together. "
                "Agent must refactor the function to support both features while passing all tests."
            ),
            base_main_py=(
                "def process(a: int, b: int) -> int:\n"
                "    return a + b\n"
            ),
            base_test_py=(
                "from app.main import process\n\n"
                "def test_process_add():\n"
                "    assert process(2, 3) == 5\n"
            ),
            feature_main_py=(
                "def process(a: int, b: int, op: str = 'multiply') -> int:\n"
                '    """Feature A: added multiply mode."""\n'
                "    if op == 'multiply':\n"
                "        return a * b\n"
                "    return a + b\n"
            ),
            main_branch_main_py=(
                "def process(a: int, b: int, op: str = 'power') -> int:\n"
                '    """Feature B: added power mode."""\n'
                "    if op == 'power':\n"
                "        return a ** b\n"
                "    return a + b\n"
            ),
            solution_main_py=(
                "def process(a: int, b: int, op: str = 'add') -> int:\n"
                '    """Supports add, multiply, and power operations."""\n'
                "    if op == 'multiply':\n"
                "        return a * b\n"
                "    elif op == 'power':\n"
                "        return a ** b\n"
                "    return a + b\n"
            ),
            final_test_py=(
                "from app.main import process\n\n"
                "def test_process_add():\n"
                "    assert process(2, 3) == 5\n"
                "    assert process(2, 3, op='add') == 5\n\n"
                "def test_process_multiply():\n"
                "    assert process(2, 3, op='multiply') == 6\n\n"
                "def test_process_power():\n"
                "    assert process(2, 3, op='power') == 8\n"
            ),
        )

    @classmethod
    def get(cls, level: str) -> TaskScenario:
        """Get a task scenario by difficulty level."""
        if not cls._tasks:
            cls._register_defaults()
        if level not in cls._tasks:
            raise ValueError(f"Unknown task level: {level}. Available: {list(cls._tasks.keys())}")
        return cls._tasks[level]

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all available task levels."""
        if not cls._tasks:
            cls._register_defaults()
        return list(cls._tasks.keys())
