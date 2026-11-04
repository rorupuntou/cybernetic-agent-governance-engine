# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Handover Relay Manager for LangGraph state transitions.

Enables safe landing under quota exhaustion and seamless warm-start
bootstrapping for successor agents based on RFC #652.
"""

from __future__ import annotations

import logging
from typing import Any

from src.gateway.client.exceptions import CageGatewayError, QuotaExceededException
from src.gateway.governance.contracts import AgentHandoverManifest

logger = logging.getLogger("CageClient.HandoverRelay")


class DirtyStateRejectionError(CageGatewayError):
    """Raised when a successor agent refuses to bootstrap over broken or dirty predecessor state."""

    pass


class HandoverRelayManager:
    """Manages the lifecycle of elastic quota transitions and warm-start relays in LangGraph."""

    @staticmethod
    def handle_quota_exhaustion(
        state: dict[str, Any],
        exception: QuotaExceededException,
    ) -> dict[str, Any]:
        """Intercepts quota exhaustion in @cage_guard and flags state for safe landing."""
        task_id = state.get("task_id", "unknown")
        logger.warning("Quota EXHAUSTED for task %s. Initiating safe landing.", task_id)

        updated_state = dict(state)
        updated_state["governance_status"] = "HANDOVER_REQUIRED"
        updated_state["quota_telemetry"] = exception.telemetry
        return updated_state

    @staticmethod
    def emit_manifest(
        state: dict[str, Any],
        system_diagnostics: dict[str, bool],
        successor_briefing: str,
    ) -> dict[str, Any]:
        """Produces an immutable AgentHandoverManifest using the safe landing reserve."""
        manifest = AgentHandoverManifest(
            task_id=state.get("task_id", ""),
            agent_id=state.get("agent_id", ""),
            status="CHECKPOINTED_AWAITING_RELAY",
            repository_clean=system_diagnostics.get("repository_clean", False),
            syntax_valid=system_diagnostics.get("syntax_valid", False),
            tests_passing=system_diagnostics.get("tests_passing", False),
            completed_milestones=tuple(state.get("completed_steps", [])),
            pending_actions=tuple(state.get("pending_plan_actions", [])),
            successor_instructions=successor_briefing,
            total_tokens_spent=state.get("accumulated_tokens", 0),
            burst_tokens_used=state.get("burst_tokens", 0),
        )

        updated_state = dict(state)
        updated_state["handover_manifest"] = manifest.to_dict()
        updated_state["governance_status"] = "RELAY_READY"
        return updated_state

    @staticmethod
    def bootstrap_successor(
        manifest_data: dict[str, Any] | AgentHandoverManifest,
        fallback_model_tier: str = "flash_lite",
    ) -> dict[str, Any]:
        """Deserializes the manifest and constructs an immediate warm-start execution state."""
        if isinstance(manifest_data, AgentHandoverManifest):
            task_id = manifest_data.task_id
            repo_clean = manifest_data.repository_clean
            syntax_valid = manifest_data.syntax_valid
            pending_actions = list(manifest_data.pending_actions)
            completed_milestones = list(manifest_data.completed_milestones)
            successor_instructions = manifest_data.successor_instructions
        else:
            task_id = manifest_data.get("task_id", "unknown")
            state_integrity = manifest_data.get("state_integrity", {})
            repo_clean = state_integrity.get(
                "repository_clean", manifest_data.get("repository_clean", False)
            )
            syntax_valid = state_integrity.get(
                "syntax_valid", manifest_data.get("syntax_valid", False)
            )
            pending_actions = list(manifest_data.get("pending_actions", []))
            completed_milestones = list(manifest_data.get("completed_milestones", []))
            successor_instructions = str(
                manifest_data.get("successor_instructions", "")
            )

        # Fail-closed enforcement: never run on dirty repository or invalid syntax
        if not repo_clean or not syntax_valid:
            raise DirtyStateRejectionError(
                f"Cannot bootstrap successor for task {task_id}. "
                f"Predecessor left a DIRTY_STATE (repository_clean={repo_clean}, syntax_valid={syntax_valid})."
            )

        if not pending_actions:
            logger.info(
                "No pending actions remaining for task %s. Evaluation terminal.",
                task_id,
            )
            return {"entry_point": "end", "governance_status": "COMPLETED"}

        next_action = pending_actions[0]
        target_node = next_action.get("node_name", "execute_action")

        logger.info(
            "Warm-starting successor agent for task %s. Routing directly to node: '%s'",
            task_id,
            target_node,
        )

        return {
            "task_id": task_id,
            "agent_id": f"successor-{task_id}",
            "model_tier_override": fallback_model_tier,
            "completed_steps": completed_milestones,
            "pending_plan_actions": pending_actions,
            "governance_status": "NORMAL",
            "entry_point": target_node,
            "system_instruction_addendum": (
                f"WARM RELAY BRIEFING:\n{successor_instructions}"
            ),
        }
