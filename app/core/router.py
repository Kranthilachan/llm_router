"""
ATLAS Router — main orchestration.

route() is the single entry point: it runs the full staged pipeline
(parse -> feasibility -> policy -> Bayesian quality -> Pareto reduction ->
utility scoring -> confidence estimation -> plan generation -> escalation
ladder) and returns a complete, auditable RoutingDecision.

record_outcome() is the learning hook: it feeds observed outcomes back
into a model's Bayesian priors so future routing improves over time.
"""

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app.config import (
    CONFIDENCE_ABSTAIN_THRESHOLD,
    CONFIDENCE_ESCALATE_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD,
    REGISTRY_VERSION,
    ROUTER_VERSION,
)
from app.core.feasibility import feasibility_filter
from app.core.planning import (
    choose_verifier_models,
    generate_multi_stage_plan,
    generate_single_model_plan,
)
from app.core.policy import apply_policy_to_models, evaluate_policy
from app.core.scoring import (
    attach_contextual_quality,
    choose_effective_profile,
    compute_utilities,
    estimate_confidence,
    pareto_frontier,
)
from app.core.semantic_parser import parse_task_request
from app.models.registry_builder import MODEL_REGISTRY
from app.models.schemas import RequestConstraints, RoutingDecision, TenantContext


def route(
    prompt: str,
    input_formats: Optional[List[str]] = None,
    estimated_tokens: int = 2000,
    estimated_output_tokens: int = 1200,
    artifact_hints: Optional[List[Dict[str, Any]]] = None,
    llm_parser: Optional[Callable] = None,
    request_constraints: Optional[RequestConstraints] = None,
    tenant_context: Optional[TenantContext] = None,
    files: Optional[List[str]] = None,
    profile_name: str = "balanced",
    registry: Optional[List[Dict[str, Any]]] = None,
    shadow_model: Optional[str] = None,
) -> RoutingDecision:
    """Main entry point for the ATLAS Router. Returns a complete RoutingDecision."""
    start_time = time.time()
    decision_id = str(uuid.uuid4())
    registry = registry if registry is not None else MODEL_REGISTRY

    # ---- 1-2. Parse the request / build canonical task representation ----
    task = parse_task_request(
        prompt=prompt,
        input_formats=input_formats,
        estimated_tokens=estimated_tokens,
        estimated_output_tokens=estimated_output_tokens,
        artifact_hints=artifact_hints,
        llm_parser=llm_parser,
        request_constraints=request_constraints,
        tenant_context=tenant_context,
        files=files,
    )

    # ---- 3. Feasibility filtering ----
    feasible, feasibility_reasons = feasibility_filter(task, registry)
    if not feasible:
        return RoutingDecision(
            selected_plan=None, abstain=True, escalate_to_human=True,
            decision_record={
                "decision_id": decision_id,
                "status": "no_feasible_models",
                "feasibility_reasons": feasibility_reasons,
                "task_summary": {
                    "primary_family": task.primary_family,
                    "domain": task.domain,
                    "required_formats": task.required_formats,
                },
                "router_version": ROUTER_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.time() - start_time) * 1000,
            },
        )

    # ---- 4. Governance policy ----
    policy = evaluate_policy(task)
    if policy.must_abstain:
        return RoutingDecision(
            selected_plan=None, abstain=True, escalate_to_human=False,
            decision_record={
                "decision_id": decision_id,
                "status": "policy_abstain",
                "deny_reason": policy.deny_reason,
                "policy_notes": policy.notes,
                "router_version": ROUTER_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.time() - start_time) * 1000,
            },
        )

    gated, policy_reasons = apply_policy_to_models(task, policy, feasible)
    if not gated:
        return RoutingDecision(
            selected_plan=None, abstain=True, escalate_to_human=True,
            decision_record={
                "decision_id": decision_id,
                "status": "no_models_after_policy",
                "feasibility_reasons": feasibility_reasons,
                "policy_reasons": policy_reasons,
                "policy_notes": policy.notes,
                "router_version": ROUTER_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.time() - start_time) * 1000,
            },
        )

    # ---- 5. Contextual Bayesian quality ----
    enriched = attach_contextual_quality(gated, task)

    # ---- 6. Pareto reduction (with top-3 floor) ----
    frontier = pareto_frontier(enriched, eps=0.02)
    if len(frontier) < 3 and len(enriched) >= 3:
        ranked = sorted(enriched, key=lambda m: m["q"]["runtime_adjusted_mean"], reverse=True)
        frontier = ranked[:3]

    # ---- 7. Utility scoring ----
    effective_profile = choose_effective_profile(task, profile_name)
    scored = compute_utilities(frontier, task, effective_profile)

    # ---- 8. Confidence estimation ----
    confidence_data = estimate_confidence(scored, task, effective_profile)

    primary_model = scored[0]
    fallback_models = scored[1:4] if len(scored) > 1 else []

    # ---- Verifier planning ----
    verifier_models = choose_verifier_models(
        all_models=feasible,
        verifier_types=policy.require_verifier_types,
        selected_model_name=primary_model["name"],
        max_verifiers=2,
    )

    # ---- 9. Execution plan generation ----
    single_plan = generate_single_model_plan(
        model=primary_model, task=task, confidence_data=confidence_data,
        verifier_models=verifier_models, policy=policy,
        fallback_models=fallback_models, profile_name=effective_profile,
    )
    plan = single_plan
    if not task.request_constraints.must_use_single_model and len(task.required_stages) >= 2:
        multi_plan = generate_multi_stage_plan(
            models=scored, task=task, confidence_data=confidence_data,
            verifier_models=verifier_models, policy=policy,
            profile_name=effective_profile,
        )
        if multi_plan and multi_plan.expected_quality >= single_plan.expected_quality + 0.03:
            plan = multi_plan

    # ---- 10. Abstention and escalation ladder ----
    abstain = False
    escalate = policy.must_escalate
    if confidence_data["top_confidence"] < CONFIDENCE_ABSTAIN_THRESHOLD:
        abstain = True
        escalate = True
    elif confidence_data["top_confidence"] < CONFIDENCE_ESCALATE_THRESHOLD:
        escalate = True
    elif confidence_data["top_confidence"] < task.request_constraints.min_confidence:
        escalate = True
    if task.risk_tier == "high" and confidence_data["top_confidence"] < CONFIDENCE_HIGH_THRESHOLD:
        escalate = True

    # ---- Shadow routing hook ----
    shadow_info = None
    if shadow_model:
        shadow_candidates = [m for m in scored if m["name"] == shadow_model]
        if shadow_candidates:
            shadow_info = {
                "shadow_model": shadow_model,
                "shadow_utility": shadow_candidates[0]["u"]["expected_utility"],
                "shadow_quality": shadow_candidates[0]["q"]["runtime_adjusted_mean"],
            }

    # ---- 11. Final decision record ----
    elapsed_ms = (time.time() - start_time) * 1000
    decision_record = {
        "decision_id": decision_id,
        "status": "routed" if not abstain else "abstained",
        "router_version": ROUTER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_ms": elapsed_ms,
        "task_summary": {
            "primary_family": task.primary_family,
            "secondary_families": task.secondary_families,
            "domain": task.domain,
            "risk_tier": task.risk_tier,
            "risk_type": task.risk_type,
            "complexity": task.complexity,
            "workflow_profile": task.workflow_profile,
            "required_stages": task.required_stages,
            "expected_output": task.expected_output,
            "requires_json": task.requires_json,
            "requires_ocr": task.requires_ocr,
            "requires_web_search": task.requires_web_search,
            "requires_citations": task.requires_citations,
            "requires_verifier": task.requires_verifier,
            "parser_confidence": task.parser_confidence,
            "conflict_flags": task.conflict_flags,
            "detected_languages": task.detected_languages,
            "total_file_size_bytes": task.total_file_size_bytes,
            "inferred_topics": [a.inferred_topic for a in task.artifacts if a.inferred_topic],
            "ambiguity_score": task.ambiguity_score,
            "decomposition_needed": task.decomposition_needed,
        },
        "pipeline_trace": {
            "registry_models": len(registry),
            "feasible_after_filter": len(feasible),
            "feasibility_exclusions": feasibility_reasons,
            "policy_exclusions": policy_reasons,
            "after_policy": len(gated),
            "after_pareto": len(frontier),
            "scored_candidates": len(scored),
            "profile_used": effective_profile,
            "policy_notes": policy.notes,
            "policy_verifier_types": policy.require_verifier_types,
        },
        "confidence": {
            "top_confidence": confidence_data["top_confidence"],
            "margin": confidence_data["margin"],
            "win_probabilities": confidence_data["win_probabilities"],
            "min_confidence_threshold": task.request_constraints.min_confidence,
        },
        "selected_plan_id": plan.plan_id if plan else None,
        "selected_plan_type": plan.plan_type if plan else None,
        "escalate_to_human": escalate,
        "abstain": abstain,
        "shadow_routing": shadow_info,
        "observability": {
            "routing_latency_ms": elapsed_ms,
            "models_evaluated": len(scored),
            "confidence_sims": confidence_data["n_simulations"],
        },
        "learning_hook": {
            "outcome_pending": True,
            "primary_model": primary_model["name"],
            "task_family": task.primary_family,
            "domain": task.domain,
            "confidence": confidence_data["top_confidence"],
            "note": "Record outcome to update priors via Bayesian learning.",
        },
    }

    hash_input = json.dumps({
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "task_family": task.primary_family,
        "domain": task.domain,
        "profile": effective_profile,
        "feasible_models": sorted([m["name"] for m in feasible]),
        "registry_version": REGISTRY_VERSION,
    }, sort_keys=True)
    decision_record["reproducibility_hash"] = hashlib.sha256(hash_input.encode()).hexdigest()[:24]

    return RoutingDecision(
        selected_plan=plan,
        abstain=abstain,
        escalate_to_human=escalate,
        decision_record=decision_record,
    )


def record_outcome(
    registry: List[Dict[str, Any]],
    model_name: str,
    task_family: str,
    success: bool,
    quality_score: float = 0.0,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    user_accepted: bool = True,
    safety_flagged: bool = False,
) -> None:
    """Update model evaluation state and Bayesian priors from an observed outcome.

    NOTE: mutates `registry` in place. Callers using the shared in-process
    MODEL_REGISTRY should hold an appropriate lock in multi-threaded/async
    contexts (see app/api/routes.py for the FastAPI-level lock).
    """
    for model in registry:
        if model["name"] == model_name:
            ev = model["evaluation"]
            ev["samples"] += 1
            ev["quality_sum"] += quality_score
            ev["latency_sum_ms"] += latency_ms
            ev["cost_sum"] += cost_usd
            ev["last_updated"] = datetime.now(timezone.utc).isoformat()
            if success:
                ev["wins"] += 1
            else:
                ev["losses"] += 1
            if user_accepted:
                ev["user_acceptance_sum"] += 1.0
            if safety_flagged:
                ev["safety_flags"] += 1
            fam_prior = model["priors"]["task_family"].get(task_family)
            if fam_prior:
                if success:
                    fam_prior["alpha"] += 1
                else:
                    fam_prior["beta"] += 1
            break
