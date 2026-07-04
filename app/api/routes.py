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
import json
from dataclasses import asdict
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

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
from app.core.execution import executable_model_names, maybe_execute_routed_answer
from app.core.formatting import format_decision_summary
from app.core.llm_domain_parser import hybrid_tiebreaker_parser
from app.core.router import record_outcome, route
from app.db.registry_repo import load_registry, update_model_outcome, add_model
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


def _apply_execution_constraints(
    rc: RequestConstraints,
    tc: TenantContext,
    *,
    execute_answer: bool,
) -> tuple[RequestConstraints, TenantContext]:
    if not execute_answer:
        return rc, tc

    executable_models = executable_model_names()
    try:
        from app.db.registry_repo import load_registry
        registry = load_registry()
        # Find all unique providers for the executable models
        executable_providers = list({m["provider"] for m in registry if m["name"] in executable_models})
    except Exception:
        executable_providers = ["OpenAI"]

    if rc.allowed_providers is None:
        rc.allowed_providers = list(executable_providers)
    else:
        filtered = [p for p in rc.allowed_providers if p in executable_providers]
        # If intersection is empty, we must enforce an impossible constraint rather than failing open
        rc.allowed_providers = filtered if filtered else ["__NONE_FEASIBLE__"]

    if tc.allowed_models is None:
        tc.allowed_models = list(executable_models)
    else:
        tc.allowed_models = [m for m in tc.allowed_models if m in executable_models]

    return rc, tc


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


_EXT_TO_INPUT_FORMAT = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".aac": "audio",
    ".txt": "text",
    ".md": "text",
}


def _normalize_uploaded_input_formats(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    if not raw.strip():
        return None
    parsed = None
    try:
        parsed_json = json.loads(raw)
        if isinstance(parsed_json, list):
            parsed = parsed_json
    except json.JSONDecodeError:
        parsed = None
    if parsed is None:
        parsed = raw.replace(";", ",").split(",")
    normalized = [item.strip().lower() for item in parsed if isinstance(item, str) and item.strip()]
    return normalized or None


def _infer_input_formats_from_uploads(files: Optional[List[UploadFile]]) -> Optional[List[str]]:
    if not files:
        return None
    inferred: List[str] = []
    seen: set = set()
    for upload in files:
        name = upload.filename or ""
        ext = Path(name).suffix.lower()
        fmt = _EXT_TO_INPUT_FORMAT.get(ext)
        if fmt and fmt not in seen:
            seen.add(fmt)
            inferred.append(fmt)
    return inferred or None


def _save_uploads(files: Optional[List[UploadFile]]) -> List[str]:
    if not files:
        return []
    saved_files: List[str] = []
    temp_root = tempfile.mkdtemp(prefix="atlas-route-upload-")
    try:
        for upload in files:
            filename = upload.filename or "upload.bin"
            safe_name = Path(filename).name
            suffix = Path(safe_name).suffix
            fd, out_path = tempfile.mkstemp(suffix=suffix, dir=temp_root)
            os.close(fd)
            try:
                with open(out_path, "wb") as dst:
                    shutil.copyfileobj(upload.file, dst)
            finally:
                upload.file.close()
            saved_files.append(out_path)
        return saved_files
    except Exception:
        for p in saved_files:
            try:
                os.unlink(p)
            except Exception:
                pass
        try:
            shutil.rmtree(temp_root, ignore_errors=True)
        except Exception:
            pass
        raise


def _cleanup_uploads(saved_paths: List[str]) -> None:
    for path in saved_paths:
        try:
            os.unlink(path)
        except Exception:
            pass
    parent = Path(saved_paths[0]).parent if saved_paths else None
    if parent is not None:
        try:
            shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            pass


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
    request_constraints = _to_request_constraints(payload.request_constraints)
    tenant_context = _to_tenant_context(payload.tenant_context)
    request_constraints, tenant_context = _apply_execution_constraints(
        request_constraints,
        tenant_context,
        execute_answer=payload.execute_answer,
    )

    try:
        async with _registry_lock:
            decision = route(
                prompt=payload.prompt,
                input_formats=payload.input_formats,
                estimated_tokens=payload.estimated_tokens,
                estimated_output_tokens=payload.estimated_output_tokens,
                artifact_hints=payload.artifact_hints,
                request_constraints=request_constraints,
                tenant_context=tenant_context,
                files=payload.files,
                profile_name=payload.profile_name,
                shadow_model=payload.shadow_model,
                registry=load_registry(),
                llm_parser=hybrid_tiebreaker_parser if LLM_PARSER_ENABLED else None,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    execution = None
    if payload.execute_answer:
        execution = maybe_execute_routed_answer(
            decision=decision,
            prompt=payload.prompt,
            input_formats=payload.input_formats,
            files=payload.files,
        )

    return RouteResponse(
        abstain=decision.abstain,
        escalate_to_human=decision.escalate_to_human,
        selected_plan=_plan_to_out(decision.selected_plan),
        decision_record=_strip_noise(decision.decision_record),
        summary_text=format_decision_summary(decision),
        answer_text=execution.answer_text if execution else None,
        execution_status=execution.status if execution else None,
        execution_metadata=execution.metadata if execution else None,
        execution_warnings=execution.warnings if execution else [],
        execution_error=execution.error if execution else None,
    )


@router.post("/route/upload", response_model=RouteResponse)
async def route_upload_request(request: Request) -> RouteResponse:
    """
    Accepts either:
      * multipart/form-data with files + form fields (execute_answer defaults to True)
      * application/json body identical to /route (execute_answer also defaults to True here)
    """
    content_type = (request.headers.get("content-type") or "").lower()

    # JSON path — same schema as /route, but execute_answer defaults to True.
    if "application/json" in content_type:
        try:
            payload_dict = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload_dict, dict):
            raise HTTPException(status_code=422, detail="JSON body must be an object.")
        payload_dict.setdefault("execute_answer", True)
        try:
            payload = RouteRequest(**payload_dict)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid request body: {exc}") from exc
        return await route_request(payload)

    # Multipart form-data path.
    form = await request.form()
    prompt_raw = form.get("prompt")
    prompt = str(prompt_raw).strip() if prompt_raw is not None else ""
    files = [f for f in form.getlist("files") if isinstance(f, UploadFile)]
    input_formats_raw = form.get("input_formats")
    input_formats = str(input_formats_raw) if input_formats_raw is not None else None

    def _form_int(name: str, default: int) -> int:
        raw = form.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return default
        try:
            return int(str(raw))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"'{name}' must be an integer.") from exc

    def _form_bool(name: str, default: bool) -> bool:
        raw = form.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return default
        return str(raw).strip().lower() in ("true", "1", "yes", "on")

    estimated_tokens = _form_int("estimated_tokens", 2000)
    estimated_output_tokens = _form_int("estimated_output_tokens", 1200)
    artifact_hints_raw = form.get("artifact_hints")
    artifact_hints = str(artifact_hints_raw) if artifact_hints_raw is not None else None
    profile_name = str(form.get("profile_name") or "balanced")
    shadow_model_raw = form.get("shadow_model")
    shadow_model = str(shadow_model_raw) if shadow_model_raw is not None else None
    execute_answer = _form_bool("execute_answer", True)

    if not prompt and not files:
        raise HTTPException(status_code=422, detail="At least one of prompt text or files is required.")

    normalized_input_formats = _normalize_uploaded_input_formats(input_formats)
    if normalized_input_formats is None:
        normalized_input_formats = _infer_input_formats_from_uploads(files)
    parsed_artifact_hints = None
    if artifact_hints:
        try:
            parsed = json.loads(artifact_hints)
            if not isinstance(parsed, list):
                raise ValueError
            parsed_artifact_hints = parsed
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="Invalid artifact_hints JSON. Provide a JSON array, e.g. [{\"format\":\"pdf\",\"page_count\":12}]",
            ) from exc

    saved_files = []
    try:
        saved_files = _save_uploads(files)

        request_constraints = _to_request_constraints(None)
        tenant_context = _to_tenant_context(None)
        request_constraints, tenant_context = _apply_execution_constraints(
            request_constraints,
            tenant_context,
            execute_answer=execute_answer,
        )

        try:
            async with _registry_lock:
                decision = route(
                    prompt=prompt,
                    input_formats=normalized_input_formats,
                    estimated_tokens=estimated_tokens,
                    estimated_output_tokens=estimated_output_tokens,
                    artifact_hints=parsed_artifact_hints,
                    request_constraints=request_constraints,
                    tenant_context=tenant_context,
                    files=saved_files,
                    profile_name=profile_name,
                    shadow_model=shadow_model,
                    registry=load_registry(),
                    llm_parser=hybrid_tiebreaker_parser if LLM_PARSER_ENABLED else None,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        execution = None
        if execute_answer:
            execution = maybe_execute_routed_answer(
                decision=decision,
                prompt=prompt,
                input_formats=normalized_input_formats,
                files=saved_files,
            )

        return RouteResponse(
            abstain=decision.abstain,
            escalate_to_human=decision.escalate_to_human,
            selected_plan=_plan_to_out(decision.selected_plan),
            decision_record=_strip_noise(decision.decision_record),
            summary_text=format_decision_summary(decision),
            answer_text=execution.answer_text if execution else None,
            execution_status=execution.status if execution else None,
            execution_metadata=execution.metadata if execution else None,
            execution_warnings=execution.warnings if execution else [],
            execution_error=execution.error if execution else None,
        )
    finally:
        if saved_files:
            _cleanup_uploads(saved_files)


@router.post("/outcome", status_code=204)
async def submit_outcome(payload: OutcomeIn) -> None:
    async with _registry_lock:
        registry = load_registry()
        names = {m["name"] for m in registry}
        if payload.model_name not in names:
            raise HTTPException(status_code=404, detail=f"Unknown model: {payload.model_name}")
        
        # We still update the in-memory copy returned by load_registry() if needed by record_outcome,
        # but since record_outcome just mutates a local list now, we will replace its call
        # with a call to update_model_outcome which writes directly to the DB.
        
        try:
            update_model_outcome(
                model_name=payload.model_name,
                task_family=payload.task_family,
                success=payload.success,
                quality_score=payload.quality_score,
                latency_ms=payload.latency_ms,
                cost_usd=payload.cost_usd,
                user_accepted=payload.user_accepted,
                safety_flagged=payload.safety_flagged,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))


@router.get("/models", response_model=List[ModelSummaryOut])
async def list_models() -> List[ModelSummaryOut]:
    registry = load_registry()
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
        for m in registry
    ]


@router.post("/models", status_code=201)
async def create_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "name" not in payload:
        raise HTTPException(status_code=422, detail="Model must have a 'name' field.")
    
    async with _registry_lock:
        try:
            model_id = add_model(payload)
            return {"status": "success", "model_id": model_id, "name": payload["name"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to insert model: {e}")

@router.get("/models/{name}")
async def get_model(name: str) -> Dict[str, Any]:
    registry = load_registry()
    for m in registry:
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
