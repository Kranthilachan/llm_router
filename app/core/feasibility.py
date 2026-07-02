"""
Feasibility & Hard Constraint Filtering.

Removes candidate models that cannot satisfy mandatory requirements —
modalities, capabilities, context window, provider/tier restrictions,
tenant allowlists, region, and budget — before any scoring happens.
A model that fails here is eliminated regardless of benchmark strength
(see Core Design Principle #1: hard constraints override soft preferences).
"""

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from app.core.scoring import estimate_request_cost_usd
from app.models.schemas import TaskFeatures


def feasibility_filter(
    task: TaskFeatures,
    registry: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    feasible = []
    reasons = {}
    rc = task.request_constraints
    tc = task.tenant_context

    for model in registry:
        name = model["name"]

        if model.get("status") != "active" or not model.get("api_available", False):
            reasons[name] = "inactive_or_no_api"
            continue
        if model["ops_dynamic"].get("incident_status") == "red":
            reasons[name] = "runtime_incident_red"
            continue
        if rc.allowed_providers and model["provider"] not in rc.allowed_providers:
            reasons[name] = "provider_not_allowed"
            continue
        if rc.disallowed_providers and model["provider"] in rc.disallowed_providers:
            reasons[name] = "provider_disallowed"
            continue
        if rc.allowed_tiers and model["tier"] not in rc.allowed_tiers:
            reasons[name] = "tier_not_allowed"
            continue
        if rc.no_open_weight and model.get("open_weight", False):
            reasons[name] = "open_weight_disallowed"
            continue
        if tc.allowed_models and name not in tc.allowed_models:
            reasons[name] = "tenant_model_not_allowed"
            continue
        if rc.required_region:
            regions = model.get("allowed_regions", [])
            if rc.required_region not in regions and "global" not in regions:
                reasons[name] = "region_not_supported"
                continue

        mods = model["modalities"]
        caps = model["capabilities"]
        ctx = model["context"]

        missing_fmt = [f for f in task.required_formats if not mods.get(f, False)]
        if missing_fmt:
            reasons[name] = f"missing_format:{','.join(missing_fmt)}"
            continue
        if task.requires_json and not caps.get("json_mode", False):
            reasons[name] = "missing_json_mode"
            continue
        if task.requires_function_calling and not caps.get("function_calling", False):
            reasons[name] = "missing_function_calling"
            continue
        if task.requires_web_search and not caps.get("web_search", False):
            reasons[name] = "missing_web_search"
            continue
        if task.requires_ocr and not caps.get("ocr", False):
            reasons[name] = "missing_ocr"
            continue
        if task.requires_citations and not caps.get("citation_support", False):
            reasons[name] = "missing_citation_support"
            continue
        if ctx["window"] < task.min_context_window:
            reasons[name] = "insufficient_context_window"
            continue

        est_cost = estimate_request_cost_usd(model, task.estimated_tokens, task.estimated_output_tokens)
        if rc.max_cost_usd is not None and est_cost > rc.max_cost_usd:
            reasons[name] = f"estimated_cost_exceeds_budget:{est_cost:.4f}"
            continue
        if tc.budget_remaining_usd is not None and est_cost > tc.budget_remaining_usd:
            reasons[name] = f"exceeds_tenant_remaining_budget:{est_cost:.4f}"
            continue

        feasible.append(deepcopy(model))

    return feasible, reasons
