"""
FastAPI routes exposing the ATLAS Router as a service.

Endpoints:
    POST /route          -> run the full routing pipeline, return a RoutingDecision
    POST /outcome         -> feed an observed outcome back into Bayesian priors
    GET  /models           -> list the canonical registry (summary view)
    GET  /models/{name}     -> full registry entry for one model
    GET  /health              -> liveness/readiness probe
    GET  /versions              -> current version stamps for every subsystem
"""

import asyncio
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ExecutionPlanOut,
    ModelSummaryOut,
    OutcomeIn,
    RequestConstraintsIn,
    RouteRequest,
    RouteResponse,
    StageRouteOut,
    TenantContextIn,
)
from app.config import (
    CALIBRATION_VERSION,
    DEFAULT_ALLOWED_PROVIDERS,
    LLM_PARSER_ENABLED,
    PARSER_VERSION,
    POLICY_VERSION,
    REGISTRY_VERSION,
    ROUTER_VERSION,
    SCORING_VERSION,
    TELEMETRY_SNAPSHOT_VERSION,
)
from app.core.formatting import format_decision_summary
from app.core.llm_domain_parser import hybrid_tiebreaker_parser
from app.core.router import record_outcome, route
from app.models.registry_builder import MODEL_REGISTRY
from app.models.schemas import RequestConstraints, TenantContext

router = APIRouter()

# The registry is mutated by /outcome (Bayesian prior updates). A single
# process-wide lock keeps concurrent requests from racing on the same
# in-memory list. For multi-worker deployments, back this with a real
# datastore instead of an in-process list.
_registry_lock = asyncio.Lock()


def _to_request_constraints(rc_in: Optional[RequestConstraintsIn]) -> RequestConstraints:
    if rc_in is None:
        rc = RequestConstraints()
    else:
        rc = RequestConstraints(**rc_in.model_dump())
    # Apply the ATLAS_ALLOWED_PROVIDERS env-var default when the caller didn't
    # set its own filter. Per-request `allowed_providers` always wins.
    if rc.allowed_providers is None and DEFAULT_ALLOWED_PROVIDERS:
        rc.allowed_providers = list(DEFAULT_ALLOWED_PROVIDERS)
    return rc


def _to_tenant_context(tc_in: Optional[TenantContextIn]) -> TenantContext:
    if tc_in is None:
        return TenantContext()
    return TenantContext(**tc_in.model_dump())


def _strip_noise(record: Dict[str, Any]) -> Dict[str, Any]:
    # Trim per-model dumps from the response — callers only need the
    # primary + fallbacks + verifiers, not the full list of rejected or
    # simulated candidates. Removes:
    #   - confidence.win_probabilities  (Thompson-Sampling per-model dict)
    #   - pipeline_trace.feasibility_exclusions (rejected models + reason)
    #   - pipeline_trace.policy_exclusions      (policy-rejected models)
    # Counts (registry_models, after_pareto, etc.) stay for observability.
    scrubbed = dict(record)
    conf = scrubbed.get("confidence")
    if isinstance(conf, dict) and "win_probabilities" in conf:
        scrubbed["confidence"] = {k: v for k, v in conf.items() if k != "win_probabilities"}
    pt = scrubbed.get("pipeline_trace")
    if isinstance(pt, dict):
        scrubbed["pipeline_trace"] = {
            k: v for k, v in pt.items()
            if k not in ("feasibility_exclusions", "policy_exclusions")
        }
    return scrubbed


def _plan_to_out(plan) -> Optional[ExecutionPlanOut]:
    if plan is None:
        return None
    trace = {k: v for k, v in plan.trace.items() if k != "win_probabilities"}
    return ExecutionPlanOut(
        plan_id=plan.plan_id,
        plan_type=plan.plan_type,
        selected_model=plan.selected_model,
        stage_routes=[StageRouteOut(**asdict(sr)) for sr in plan.stage_routes],
        fallback_models=plan.fallback_models,
        verifier_models=plan.verifier_models,
        expected_latency_ms=plan.expected_latency_ms,
        expected_cost_usd=plan.expected_cost_usd,
        expected_quality=plan.expected_quality,
        confidence=plan.confidence,
        utility=plan.utility,
        confidence_margin=plan.confidence_margin,
        profile_used=plan.profile_used,
        explanation=plan.explanation,
        trace=trace,
    )


@router.post("/route", response_model=RouteResponse)
async def route_request(payload: RouteRequest) -> RouteResponse:
    try:
        async with _registry_lock:
            decision = route(
                prompt=payload.prompt,
                input_formats=payload.input_formats,
                estimated_tokens=payload.estimated_tokens,
                estimated_output_tokens=payload.estimated_output_tokens,
                artifact_hints=payload.artifact_hints,
                request_constraints=_to_request_constraints(payload.request_constraints),
                tenant_context=_to_tenant_context(payload.tenant_context),
                files=payload.files,
                profile_name=payload.profile_name,
                shadow_model=payload.shadow_model,
                registry=MODEL_REGISTRY,
                llm_parser=hybrid_tiebreaker_parser if LLM_PARSER_ENABLED else None,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RouteResponse(
        abstain=decision.abstain,
        escalate_to_human=decision.escalate_to_human,
        selected_plan=_plan_to_out(decision.selected_plan),
        decision_record=_strip_noise(decision.decision_record),
        summary_text=format_decision_summary(decision),
    )


@router.post("/outcome", status_code=204)
async def submit_outcome(payload: OutcomeIn) -> None:
    async with _registry_lock:
        names = {m["name"] for m in MODEL_REGISTRY}
        if payload.model_name not in names:
            raise HTTPException(status_code=404, detail=f"Unknown model: {payload.model_name}")
        record_outcome(
            registry=MODEL_REGISTRY,
            model_name=payload.model_name,
            task_family=payload.task_family,
            success=payload.success,
            quality_score=payload.quality_score,
            latency_ms=payload.latency_ms,
            cost_usd=payload.cost_usd,
            user_accepted=payload.user_accepted,
            safety_flagged=payload.safety_flagged,
        )


@router.get("/models", response_model=List[ModelSummaryOut])
async def list_models() -> List[ModelSummaryOut]:
    return [
        ModelSummaryOut(
            name=m["name"],
            provider=m["provider"],
            tier=m["tier"],
            status=m["status"],
            open_weight=m["open_weight"],
            context_window=m["context"]["window"],
            relative_cost_score=m["pricing"]["relative_cost_score"],
            incident_status=m["ops_dynamic"]["incident_status"],
        )
        for m in MODEL_REGISTRY
    ]


@router.get("/models/{name}")
async def get_model(name: str) -> Dict[str, Any]:
    for m in MODEL_REGISTRY:
        if m["name"] == name:
            return m
    raise HTTPException(status_code=404, detail=f"Unknown model: {name}")


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/versions")
async def versions() -> Dict[str, str]:
    return {
        "router_version": ROUTER_VERSION,
        "parser_version": PARSER_VERSION,
        "policy_version": POLICY_VERSION,
        "scoring_version": SCORING_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "registry_version": REGISTRY_VERSION,
        "telemetry_snapshot_version": TELEMETRY_SNAPSHOT_VERSION,
    }
