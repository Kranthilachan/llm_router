from typing import Optional

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ExecutionPlan, RoutingDecision, StageRoute


client = TestClient(app)


def _decision(
    *,
    selected_model: Optional[str] = "o3",
    abstain: bool = False,
    escalate: bool = False,
    plan_type: str = "single_model",
) -> RoutingDecision:
    plan = None
    if selected_model is not None:
        plan = ExecutionPlan(
            plan_id="plan-123",
            plan_type=plan_type,
            selected_model=selected_model if plan_type == "single_model" else None,
            stage_routes=[
                StageRoute(
                    stage_id=1,
                    stage_name="chat",
                    selected_model=selected_model,
                    fallback_models=[],
                    verifier_models=[],
                    stage_confidence=0.62,
                    expected_latency_ms=100.0,
                    expected_cost_usd=0.01,
                    explanation="test stage",
                )
            ],
            fallback_models=[],
            verifier_models=[],
            expected_latency_ms=100.0,
            expected_cost_usd=0.01,
            expected_quality=0.9,
            confidence=0.62,
            utility=0.5,
            confidence_margin=0.05,
            profile_used="balanced",
            explanation={"primary_model": selected_model},
            trace={},
        )
    return RoutingDecision(
        selected_plan=plan,
        abstain=abstain,
        escalate_to_human=escalate,
        decision_record={"task_summary": {"primary_family": "chat"}, "status": "routed"},
    )


def test_route_without_execute_flag_skips_execution(monkeypatch):
    monkeypatch.setattr("app.api.routes.route", lambda **_: _decision())

    def _should_not_run(**kwargs):
        raise AssertionError("execution should not run")

    monkeypatch.setattr("app.api.routes.maybe_execute_routed_answer", _should_not_run)

    response = client.post("/route", json={"prompt": "Write a haiku.", "input_formats": ["text"]})
    data = response.json()

    assert response.status_code == 200
    assert data["answer_text"] is None
    assert data["execution_status"] is None
    assert data["execution_metadata"] is None
    assert data["execution_warnings"] == []
    assert data["execution_error"] is None


def test_route_execute_answer_returns_completion(monkeypatch):
    monkeypatch.setattr("app.api.routes.route", lambda **_: _decision(selected_model="o3"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _fake_call_openai(*, api_model: str, prompt: str):
        assert api_model == "o3"
        assert prompt == "Write a haiku."
        return "calm sea at sunrise", {"total_tokens": 12}, "stop"

    monkeypatch.setattr("app.core.execution._call_openai", _fake_call_openai)

    response = client.post(
        "/route",
        json={"prompt": "Write a haiku.", "input_formats": ["text"], "execute_answer": True},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["execution_status"] == "completed"
    assert data["answer_text"] == "calm sea at sunrise"
    assert data["execution_metadata"]["provider"] == "OpenAI"
    assert data["execution_metadata"]["routed_model"] == "o3"
    assert data["execution_metadata"]["api_model"] == "o3"
    assert data["execution_error"] is None


def test_route_execute_answer_skips_escalated_decision(monkeypatch):
    monkeypatch.setattr("app.api.routes.route", lambda **_: _decision(escalate=True))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _should_not_call_openai(**kwargs):
        raise AssertionError("provider call should not happen")

    monkeypatch.setattr("app.core.execution._call_openai", _should_not_call_openai)

    response = client.post(
        "/route",
        json={"prompt": "Review this contract.", "input_formats": ["text"], "execute_answer": True},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["execution_status"] == "escalated"
    assert data["answer_text"] is None
    assert data["execution_warnings"]


def test_route_execute_answer_rejects_non_text_requests(monkeypatch):
    monkeypatch.setattr("app.api.routes.route", lambda **_: _decision(selected_model="o3"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _should_not_call_openai(**kwargs):
        raise AssertionError("provider call should not happen")

    monkeypatch.setattr("app.core.execution._call_openai", _should_not_call_openai)

    response = client.post(
        "/route",
        json={"prompt": "Summarize this PDF.", "input_formats": ["pdf"], "execute_answer": True},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["execution_status"] == "unsupported"
    assert "text-only" in data["execution_warnings"][0]


def test_route_execute_answer_applies_executable_model_constraints(monkeypatch):
    captured = {}

    def _fake_route(**kwargs):
        captured["allowed_providers"] = kwargs["request_constraints"].allowed_providers
        captured["allowed_models"] = kwargs["tenant_context"].allowed_models
        return _decision(selected_model="o3")

    monkeypatch.setattr("app.api.routes.route", _fake_route)

    response = client.post(
        "/route",
        json={"prompt": "Draft a sprint plan.", "input_formats": ["text"], "execute_answer": True},
    )

    assert response.status_code == 200
    assert captured["allowed_providers"] == ["OpenAI"]
    assert "o3" in captured["allowed_models"]
    assert "GPT-4o" in captured["allowed_models"]


def test_route_execute_answer_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr("app.api.routes.route", lambda **_: _decision(selected_model="o3"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post(
        "/route",
        json={"prompt": "Write a haiku.", "input_formats": ["text"], "execute_answer": True},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["execution_status"] == "failed"
    assert data["execution_error"]["code"] == "missing_openai_api_key"
