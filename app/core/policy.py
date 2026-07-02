"""
Governance and Safety — the Policy Engine.

Policy is deliberately kept independent of scoring (Core Design Principle
#2): governance rules never hide inside utility weights. evaluate_policy()
decides *whether* a request is allowed, whether it must escalate/abstain,
and what tier/provider/verifier restrictions apply — all before any model
is scored. apply_policy_to_models() then gates the feasible candidate set
against that decision.
"""

from typing import Any, Dict, List, Tuple

from app.core.scoring import risk_support
from app.models.schemas import PolicyDecision, TaskFeatures


def evaluate_policy(task: TaskFeatures) -> PolicyDecision:
    notes: List[str] = []
    restricted_to_tiers: List[str] = []
    restricted_to_models: List[str] = []
    restricted_to_providers: List[str] = []
    require_verifier_types: List[str] = []
    overlay = task.tenant_context.policy_overlay or {}

    # Medical high-actionability escalation
    if task.domain == "medical" and task.actionability == "high":
        notes.append("Medical high-actionability task requires mandatory escalation.")
        require_verifier_types.extend(["safety_review", "factuality_review"])
        return PolicyDecision(
            allowed=True, must_escalate=True, must_abstain=False,
            restricted_to_tiers=["Frontier"],
            require_verifier_types=sorted(set(require_verifier_types)),
            notes=notes,
        )

    # Legal tasks
    if task.domain == "legal":
        notes.append("Legal task requires policy scrutiny and verification.")
        require_verifier_types.extend(["safety_review", "schema_validation"])
        restricted_to_tiers.append("Frontier")

    # Finance investment tasks
    if task.domain == "finance" and any(k in task.raw_prompt.lower() for k in ["investment", "trade", "portfolio"]):
        notes.append("Investment finance task restricted to frontier models.")
        restricted_to_tiers.append("Frontier")
        require_verifier_types.extend(["factuality_review", "safety_review"])

    # Offensive security denial
    if task.domain == "security" and any(k in task.raw_prompt.lower() for k in ["exploit", "offensive", "payload"]):
        return PolicyDecision(
            allowed=False, must_escalate=False, must_abstain=True,
            deny_reason="restricted_security_content",
            notes=["Offensive security request denied by policy."],
        )

    # General verifier requirements
    if task.requires_verifier:
        vtype = "schema_validation" if task.expected_output == "structured_json" else "factuality_review"
        require_verifier_types.append(vtype)
    if task.requires_citations:
        require_verifier_types.append("citation_review")

    # High risk tier
    if task.risk_tier == "high":
        notes.append("High-risk task restricted to frontier-grade models.")
        restricted_to_tiers.append("Frontier")

    # Tenant overlays
    if overlay.get("frontier_only", False):
        restricted_to_tiers.append("Frontier")
        notes.append("Tenant overlay enforces frontier-only routing.")
    if overlay.get("allowed_providers"):
        restricted_to_providers.extend(overlay["allowed_providers"])
        notes.append("Tenant overlay restricts providers.")
    if overlay.get("mandatory_verifier", False):
        require_verifier_types.append("consistency_review")
        notes.append("Tenant overlay requires mandatory verification.")

    return PolicyDecision(
        allowed=True, must_escalate=False, must_abstain=False,
        restricted_to_tiers=sorted(set(restricted_to_tiers)),
        restricted_to_models=sorted(set(restricted_to_models)),
        restricted_to_providers=sorted(set(restricted_to_providers)),
        require_verifier_types=sorted(set(require_verifier_types)),
        notes=notes,
    )


def apply_policy_to_models(
    task: TaskFeatures,
    policy: PolicyDecision,
    models: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    if not policy.allowed:
        return [], {m["name"]: policy.deny_reason or "policy_denied" for m in models}

    gated = []
    reasons = {}
    for model in models:
        name = model["name"]
        rs = risk_support(model, task)
        if task.risk_tier == "high" and rs < 0.85:
            reasons[name] = f"insufficient_safety:{rs:.2f}"
            continue
        if task.risk_tier == "high" and "high_risk" in model["routing"].get("avoid_for", []):
            reasons[name] = "routing_avoid_high_risk"
            continue
        if policy.restricted_to_tiers and model["tier"] not in policy.restricted_to_tiers:
            reasons[name] = f"policy_tier_restriction:{model['tier']}"
            continue
        if policy.restricted_to_models and name not in policy.restricted_to_models:
            reasons[name] = "not_in_policy_model_allowlist"
            continue
        if policy.restricted_to_providers and model["provider"] not in policy.restricted_to_providers:
            reasons[name] = "not_in_policy_provider_allowlist"
            continue
        gated.append(model)
    return gated, reasons
