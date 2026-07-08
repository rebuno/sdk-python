from rebuno.types import Execution, ExecutionStatus, Step, StepDecision


def test_execution_parse():
    e = Execution.model_validate({"id": "e1", "agent_id": "a", "input": {"x": 1}, "status": "running", "output": None})
    assert e.id == "e1"
    assert e.status is ExecutionStatus.RUNNING
    assert e.input == {"x": 1}


def test_step_decision_variants():
    d = StepDecision.model_validate({"decision": "replay", "result": {"ok": True}})
    assert d.decision == "replay"
    assert d.result == {"ok": True}
    d2 = StepDecision.model_validate({"decision": "blocked", "approval_id": "ap1"})
    assert d2.approval_id == "ap1"


def test_step_parse():
    s = Step.model_validate({"step_id": "s1", "kind": "tool_call", "status": "succeeded", "result": 7})
    assert s.step_id == "s1"
    assert s.status == "succeeded"
    assert s.result == 7


def test_extra_keys_ignored():
    # tolerate capitalized/extra keys during kernel transition
    e = Execution.model_validate({"id": "e1", "agent_id": "a", "status": "pending", "Extra": 9})
    assert e.id == "e1"
