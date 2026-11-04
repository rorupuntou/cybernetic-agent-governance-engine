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
Unit tests for LangGraph Handover Relay Adapter and Manager.

Tests cover:
1. handle_quota_exhaustion state transition
2. emit_manifest contract and payload generation
3. bootstrap_successor warm-start state creation
4. DirtyStateRejectionError fail-closed protection
5. Terminal evaluation when no pending actions remain
"""

from __future__ import annotations

import pytest

from src.gateway.client.adapters.handover_relay import (
    DirtyStateRejectionError,
    HandoverRelayManager,
    QuotaExceededException,
)
from src.gateway.governance.contracts import AgentHandoverManifest

pytestmark = [pytest.mark.unit, pytest.mark.local]


# ---------------------------------------------------------------------------
# 1. Quota Exhaustion Handling
# ---------------------------------------------------------------------------


def test_handle_quota_exhaustion_transitions_state() -> None:
    """handle_quota_exhaustion flags state as HANDOVER_REQUIRED with telemetry."""
    initial_state = {
        "task_id": "task-777",
        "agent_id": "agent-lead",
        "governance_status": "NORMAL",
        "context": {"attempt": 1},
    }
    telemetry = {
        "quota_status": "EXHAUSTED",
        "tokens_remaining": 0,
        "burst_available": False,
        "accumulated_tokens": 10500,
    }
    exc = QuotaExceededException("Token quota exceeded", telemetry=telemetry)

    updated_state = HandoverRelayManager.handle_quota_exhaustion(initial_state, exc)

    assert updated_state["governance_status"] == "HANDOVER_REQUIRED"
    assert updated_state["quota_telemetry"] == telemetry
    assert updated_state["task_id"] == "task-777"
    # Ensure immutability: initial_state is not modified in-place
    assert initial_state["governance_status"] == "NORMAL"


# ---------------------------------------------------------------------------
# 2. Manifest Emission
# ---------------------------------------------------------------------------


def test_emit_manifest_generates_valid_handover_payload() -> None:
    """emit_manifest packages state and diagnostics into a canonical manifest dictionary."""
    state = {
        "task_id": "task-888",
        "agent_id": "agent-primary",
        "completed_steps": ["lint_check", "ast_validation"],
        "pending_plan_actions": [
            {"action": "deploy_canary", "node_name": "canary_node"}
        ],
        "accumulated_tokens": 9200,
        "burst_tokens": 1200,
    }
    diagnostics = {
        "repository_clean": True,
        "syntax_valid": True,
        "tests_passing": True,
    }
    briefing = "Deploy canary stage 1, observe metrics for 60 seconds."

    result_state = HandoverRelayManager.emit_manifest(
        state=state,
        system_diagnostics=diagnostics,
        successor_briefing=briefing,
    )

    assert result_state["governance_status"] == "RELAY_READY"
    manifest = result_state["handover_manifest"]

    assert manifest["task_id"] == "task-888"
    assert manifest["agent_id"] == "agent-primary"
    assert manifest["status"] == "CHECKPOINTED_AWAITING_RELAY"
    assert manifest["state_integrity"]["repository_clean"] is True
    assert manifest["state_integrity"]["syntax_valid"] is True
    assert manifest["state_integrity"]["tests_passing"] is True
    assert manifest["completed_milestones"] == ["lint_check", "ast_validation"]
    assert manifest["pending_actions"] == [
        {"action": "deploy_canary", "node_name": "canary_node"}
    ]
    assert manifest["successor_instructions"] == briefing
    assert manifest["tokenomics"]["total_tokens_spent"] == 9200
    assert manifest["tokenomics"]["burst_tokens_used"] == 1200


# ---------------------------------------------------------------------------
# 3. Successor Bootstrapping (Warm-Start)
# ---------------------------------------------------------------------------


def test_bootstrap_successor_creates_warm_start_state_from_dict() -> None:
    """bootstrap_successor constructs surgical entry point and system briefing from dict."""
    manifest_dict = {
        "task_id": "task-999",
        "agent_id": "agent-old",
        "status": "CHECKPOINTED_AWAITING_RELAY",
        "state_integrity": {
            "repository_clean": True,
            "syntax_valid": True,
            "tests_passing": True,
        },
        "completed_milestones": ["step_1_setup", "step_2_compile"],
        "pending_actions": [
            {"action": "run_integration", "node_name": "integration_test_node"}
        ],
        "successor_instructions": "Resume directly at integration_test_node.",
    }

    successor_state = HandoverRelayManager.bootstrap_successor(
        manifest_data=manifest_dict,
        fallback_model_tier="flash_lite",
    )

    assert successor_state["task_id"] == "task-999"
    assert successor_state["agent_id"] == "successor-task-999"
    assert successor_state["model_tier_override"] == "flash_lite"
    assert successor_state["governance_status"] == "NORMAL"
    assert successor_state["entry_point"] == "integration_test_node"
    assert successor_state["completed_steps"] == ["step_1_setup", "step_2_compile"]
    assert successor_state["pending_plan_actions"] == [
        {"action": "run_integration", "node_name": "integration_test_node"}
    ]
    assert "WARM RELAY BRIEFING:" in successor_state["system_instruction_addendum"]
    assert (
        "Resume directly at integration_test_node."
        in successor_state["system_instruction_addendum"]
    )


def test_bootstrap_successor_from_dataclass_instance() -> None:
    """bootstrap_successor handles direct AgentHandoverManifest dataclass instance."""
    manifest = AgentHandoverManifest(
        task_id="task-1000",
        agent_id="agent-prev",
        status="CHECKPOINTED_AWAITING_RELAY",
        repository_clean=True,
        syntax_valid=True,
        tests_passing=True,
        completed_milestones=("step_1",),
        pending_actions=({"node_name": "node_audit", "step": 2},),
        successor_instructions="Audit previous step.",
    )

    successor_state = HandoverRelayManager.bootstrap_successor(manifest)

    assert successor_state["task_id"] == "task-1000"
    assert successor_state["entry_point"] == "node_audit"
    assert successor_state["completed_steps"] == ["step_1"]


# ---------------------------------------------------------------------------
# 4. Fail-Closed Dirty State Protection
# ---------------------------------------------------------------------------


def test_bootstrap_successor_rejects_dirty_repository() -> None:
    """bootstrap_successor raises DirtyStateRejectionError when repository is dirty."""
    dirty_manifest = {
        "task_id": "task-uncommitted",
        "state_integrity": {
            "repository_clean": False,
            "syntax_valid": True,
            "tests_passing": True,
        },
        "pending_actions": [{"node_name": "action_node"}],
    }

    with pytest.raises(DirtyStateRejectionError) as exc_info:
        HandoverRelayManager.bootstrap_successor(dirty_manifest)

    assert "DIRTY_STATE" in str(exc_info.value)
    assert "task-uncommitted" in str(exc_info.value)


def test_bootstrap_successor_rejects_invalid_syntax() -> None:
    """bootstrap_successor raises DirtyStateRejectionError when syntax is invalid."""
    invalid_syntax_manifest = {
        "task_id": "task-broken-syntax",
        "state_integrity": {
            "repository_clean": True,
            "syntax_valid": False,
            "tests_passing": False,
        },
        "pending_actions": [{"node_name": "action_node"}],
    }

    with pytest.raises(DirtyStateRejectionError) as exc_info:
        HandoverRelayManager.bootstrap_successor(invalid_syntax_manifest)

    assert "DIRTY_STATE" in str(exc_info.value)
    assert "task-broken-syntax" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Terminal State (No Pending Actions)
# ---------------------------------------------------------------------------


def test_bootstrap_successor_routes_to_end_when_no_pending_actions() -> None:
    """When no pending actions remain, bootstrap_successor sets entry_point to 'end'."""
    completed_manifest = {
        "task_id": "task-done",
        "state_integrity": {
            "repository_clean": True,
            "syntax_valid": True,
            "tests_passing": True,
        },
        "completed_milestones": ["task_a", "task_b"],
        "pending_actions": [],
    }

    result = HandoverRelayManager.bootstrap_successor(completed_manifest)

    assert result["entry_point"] == "end"
    assert result["governance_status"] == "COMPLETED"
