"""Client for the typed Git-CI-Gym repo repair environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import RepoRepairAction, RepoRepairObservation, RepoRepairState


class GitCIEnv(EnvClient[RepoRepairAction, RepoRepairObservation, RepoRepairState]):
    """Persistent WebSocket client for Git-CI-Gym."""

    def _step_payload(self, action: RepoRepairAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[RepoRepairObservation]:
        obs_data = payload.get("observation", {})
        observation = RepoRepairObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
                "metadata": obs_data.get("metadata", {}),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> RepoRepairState:
        return RepoRepairState.model_validate(payload)
