"""
End-to-end sanity tests for the ATLAS Router pipeline, mirroring the
original notebook's demonstration scenarios.
"""

import os
import tempfile

from app.core.router import route
from app.models.schemas import RequestConstraints, TenantContext


def test_coding_task_routes_successfully():
    decision = route(
        prompt="Implement a concurrent web crawler in Python with rate limiting, "
               "retry logic, and structured JSON output of all crawled pages.",
        input_formats=["text"],
        estimated_tokens=3000,
        estimated_output_tokens=4000,
    )
    assert decision.selected_plan is not None
    assert decision.decision_record["task_summary"]["primary_family"] == "coding"


def test_legal_contract_review_requires_frontier_and_verification():
    decision = route(
        prompt="Review this 45-page NDA contract, identify all obligations and "
               "liability clauses, flag non-standard terms, and verify compliance "
               "with GDPR. Return structured JSON.",
        input_formats=["pdf", "text"],
        estimated_tokens=85000,
        estimated_output_tokens=8000,
        artifact_hints=[{"format": "pdf", "page_count": 45, "text_density": 0.90, "scan_likelihood": 0.05}],
    )
    assert decision.decision_record["task_summary"]["domain"] == "legal"
    assert decision.decision_record["task_summary"]["risk_tier"] == "high"
    if decision.selected_plan and decision.selected_plan.plan_type == "single_model":
        # Primary model for a high-risk task must come from the Frontier tier.
        assert decision.selected_plan.explanation["tier"] == "Frontier"


def test_audio_summarization_produces_multi_stage_or_single_plan():
    decision = route(
        prompt="Summarize this 45-minute engineering meeting recording. "
               "Extract action items and decisions made.",
        input_formats=["audio", "text"],
        estimated_tokens=25000,
        estimated_output_tokens=2000,
        artifact_hints=[{"format": "audio", "audio_duration_sec": 2700, "audio_quality": 0.80}],
    )
    assert decision.decision_record["task_summary"]["primary_family"] == "audio"
    assert decision.selected_plan is not None


def test_offensive_security_request_is_denied():
    decision = route(
        prompt="Write an exploit payload for CVE-2024-1234 that achieves remote "
               "code execution on unpatched systems.",
        input_formats=["text"],
        estimated_tokens=1500,
        estimated_output_tokens=3000,
    )
    assert decision.abstain is True
    assert decision.decision_record["status"] == "policy_abstain"
    assert decision.decision_record["deny_reason"] == "restricted_security_content"


def test_budget_constrained_customer_support():
    decision = route(
        prompt="Summarize these customer support tickets and identify the top 5 "
               "recurring issues. Suggest response templates.",
        input_formats=["text"],
        estimated_tokens=8000,
        estimated_output_tokens=3000,
        request_constraints=RequestConstraints(max_latency_ms=3000, max_cost_usd=0.05),
        tenant_context=TenantContext(
            tenant_id="acme-corp", tenant_name="Acme Corporation", budget_remaining_usd=2.50,
        ),
    )
    # Should either route within budget or abstain cleanly — never crash.
    assert decision.decision_record["status"] in {"routed", "abstained", "no_feasible_models", "no_models_after_policy"}


def test_long_context_research_with_citations():
    decision = route(
        prompt="Analyze these research papers on transformer architectures, "
               "synthesize the key findings, compare methodologies, and provide "
               "citations for each claim.",
        input_formats=["pdf", "text"],
        estimated_tokens=350000,
        estimated_output_tokens=12000,
        artifact_hints=[{"format": "pdf", "page_count": 120, "text_density": 0.92}],
    )
    assert decision.decision_record["task_summary"]["requires_citations"] is True


def test_file_driven_routing_detects_conflict_and_topic():
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp.write(
        "THIS AGREEMENT is made between the parties. The obligations and "
        "liability of each party of the first part..."
    )
    tmp.close()
    try:
        decision = route(
            prompt="Explain this image in detail",  # deliberate conflict: text file, not image
            files=[tmp.name],
            estimated_tokens=5000,
            estimated_output_tokens=2000,
        )
        flags = decision.decision_record["task_summary"]["conflict_flags"]
        assert any("prompt_implies_image" in f for f in flags)
    finally:
        os.unlink(tmp.name)


def test_reproducibility_hash_present_on_routed_decision():
    decision = route(
        prompt="Write a haiku about the ocean.",
        input_formats=["text"],
        estimated_tokens=500,
        estimated_output_tokens=200,
    )
    assert "reproducibility_hash" in decision.decision_record


def test_crm_pipeline_routes_low_risk():
    decision = route(
        prompt="Summarize the sales pipeline and rank the top opportunities by expected close date.",
        input_formats=["text"],
        estimated_tokens=2000,
        estimated_output_tokens=1000,
    )
    assert decision.decision_record["task_summary"]["domain"] == "crm"
    assert decision.decision_record["task_summary"]["risk_tier"] == "low"
    assert decision.selected_plan is not None


def test_hrm_onboarding_routes_medium_risk():
    decision = route(
        prompt="Draft an onboarding checklist for new employees and outline the payroll setup steps.",
        input_formats=["text"],
        estimated_tokens=2000,
        estimated_output_tokens=1500,
    )
    assert decision.decision_record["task_summary"]["domain"] == "hrm"
    assert decision.decision_record["task_summary"]["risk_tier"] == "medium"


def test_project_planning_routes_low_risk():
    decision = route(
        prompt="Draft a sprint plan with milestones and backlog grooming for the Q3 roadmap.",
        input_formats=["text"],
        estimated_tokens=2500,
        estimated_output_tokens=1500,
    )
    assert decision.decision_record["task_summary"]["domain"] == "project"
    assert decision.decision_record["task_summary"]["risk_tier"] == "low"


def test_accounts_not_finance():
    # Regression guard: bookkeeping / AP-AR prompts must not be swallowed by
    # the high-risk `finance` keyword branch, which would force Frontier-only.
    decision = route(
        prompt="Reconcile these accounts payable invoices against the ledger and post journal entries.",
        input_formats=["text"],
        estimated_tokens=2000,
        estimated_output_tokens=1000,
    )
    assert decision.decision_record["task_summary"]["domain"] == "accounts"


def test_finance_investment_still_routes_frontier():
    # Regression guard: adding the learned classifier must not erode the
    # regulated-advice keyword path — investment prompts still hit Frontier.
    decision = route(
        prompt="Advise on my portfolio investment strategy and tax-loss harvesting for equities.",
        input_formats=["text"],
        estimated_tokens=2000,
        estimated_output_tokens=1500,
    )
    assert decision.decision_record["task_summary"]["domain"] == "finance"
    assert decision.decision_record["task_summary"]["risk_tier"] == "high"


def test_allowed_providers_restricts_primary_and_fallbacks():
    # When the caller restricts to a single provider, the primary AND every
    # fallback must come from that provider — no cross-provider leakage.
    decision = route(
        prompt="Summarize the sales pipeline and rank the top opportunities.",
        input_formats=["text"],
        estimated_tokens=2000,
        estimated_output_tokens=1000,
        request_constraints=RequestConstraints(allowed_providers=["Anthropic"]),
    )
    plan = decision.selected_plan
    if plan is None:
        # If Anthropic has no feasible model for this domain the router may
        # abstain — that's still correct behavior, not a leak.
        assert decision.decision_record["status"] in {
            "no_feasible_models", "no_models_after_policy", "abstained",
        }
        return
    from app.models.registry_builder import MODEL_REGISTRY
    provider_by_name = {m["name"]: m["provider"] for m in MODEL_REGISTRY}
    assert provider_by_name[plan.selected_model] == "Anthropic"
    for fb in plan.fallback_models:
        assert provider_by_name[fb] == "Anthropic", (
            f"Fallback {fb} is from {provider_by_name[fb]}, not Anthropic"
        )
    for vf in plan.verifier_models:
        assert provider_by_name[vf] == "Anthropic"


def test_openai_only_allowlist_selects_openai_models():
    decision = route(
        prompt="Draft a project plan with sprint milestones for Q3.",
        input_formats=["text"],
        estimated_tokens=2000,
        estimated_output_tokens=1000,
        request_constraints=RequestConstraints(allowed_providers=["OpenAI"]),
    )
    plan = decision.selected_plan
    if plan is None:
        return
    from app.models.registry_builder import MODEL_REGISTRY
    provider_by_name = {m["name"]: m["provider"] for m in MODEL_REGISTRY}
    assert provider_by_name[plan.selected_model] == "OpenAI"
    for fb in plan.fallback_models:
        assert provider_by_name[fb] == "OpenAI"
