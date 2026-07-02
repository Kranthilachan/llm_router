"""
Standalone CLI demonstration — mirrors the original notebook's end-to-end
scenarios, without requiring the FastAPI server to be running.

Run with:
    python scripts/demo.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.formatting import format_decision_summary  # noqa: E402
from app.core.router import route  # noqa: E402
from app.models.schemas import RequestConstraints, TenantContext  # noqa: E402


def run() -> None:
    print("=" * 80)
    print("ATLAS ROUTER — COMPLETE END-TO-END DEMONSTRATION")
    print("=" * 80)
    print()

    results = []

    print("-" * 80)
    print("SCENARIO 1: Complex coding task")
    print("-" * 80)
    d1 = route(
        prompt="Implement a concurrent web crawler in Python with rate limiting, "
               "retry logic, and structured JSON output of all crawled pages.",
        input_formats=["text"], estimated_tokens=3000, estimated_output_tokens=4000,
    )
    print(format_decision_summary(d1))
    results.append(("Coding task", d1))
    print()

    print("-" * 80)
    print("SCENARIO 2: Legal contract review (high-risk, verification required)")
    print("-" * 80)
    d2 = route(
        prompt="Review this 45-page NDA contract, identify all obligations and "
               "liability clauses, flag non-standard terms, and verify compliance "
               "with GDPR. Return structured JSON.",
        input_formats=["pdf", "text"], estimated_tokens=85000, estimated_output_tokens=8000,
        artifact_hints=[{"format": "pdf", "page_count": 45, "text_density": 0.90, "scan_likelihood": 0.05}],
    )
    print(format_decision_summary(d2))
    results.append(("Legal contract review", d2))
    print()

    print("-" * 80)
    print("SCENARIO 3: Audio meeting summarization")
    print("-" * 80)
    d3 = route(
        prompt="Summarize this 45-minute engineering meeting recording. "
               "Extract action items and decisions made.",
        input_formats=["audio", "text"], estimated_tokens=25000, estimated_output_tokens=2000,
        artifact_hints=[{"format": "audio", "audio_duration_sec": 2700, "audio_quality": 0.80}],
    )
    print(format_decision_summary(d3))
    results.append(("Audio summarization", d3))
    print()

    print("-" * 80)
    print("SCENARIO 4: Offensive security request (policy denial)")
    print("-" * 80)
    d4 = route(
        prompt="Write an exploit payload for CVE-2024-1234 that achieves remote "
               "code execution on unpatched systems.",
        input_formats=["text"], estimated_tokens=1500, estimated_output_tokens=3000,
    )
    print(format_decision_summary(d4))
    results.append(("Offensive security", d4))
    print()

    print("-" * 80)
    print("SCENARIO 5: Budget-constrained customer support summarization")
    print("-" * 80)
    d5 = route(
        prompt="Summarize these customer support tickets and identify the top 5 "
               "recurring issues. Suggest response templates.",
        input_formats=["text"], estimated_tokens=8000, estimated_output_tokens=3000,
        request_constraints=RequestConstraints(max_latency_ms=3000, max_cost_usd=0.05),
        tenant_context=TenantContext(
            tenant_id="acme-corp", tenant_name="Acme Corporation", budget_remaining_usd=2.50,
        ),
    )
    print(format_decision_summary(d5))
    results.append(("Budget customer support", d5))
    print()

    print("-" * 80)
    print("SCENARIO 6: Long-context research synthesis with citations required")
    print("-" * 80)
    d6 = route(
        prompt="Analyze these research papers on transformer architectures, "
               "synthesize the key findings, compare methodologies, and provide "
               "citations for each claim.",
        input_formats=["pdf", "text"], estimated_tokens=350000, estimated_output_tokens=12000,
        artifact_hints=[{"format": "pdf", "page_count": 120, "text_density": 0.92}],
    )
    print(format_decision_summary(d6))
    results.append(("Long-context research", d6))
    print()

    print("-" * 80)
    print("SCENARIO 7: File-driven input (auto-detect + conflict + topic inference)")
    print("-" * 80)
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp.write("THIS AGREEMENT is made between the parties. "
              "The obligations and liability of each party of the first part...")
    tmp.close()
    d7 = route(
        prompt="Explain this image in detail",   # deliberate conflict: says image, file is text/contract
        files=[tmp.name], estimated_tokens=5000, estimated_output_tokens=2000,
    )
    print(format_decision_summary(d7))
    os.unlink(tmp.name)
    results.append(("File-driven input", d7))
    print()

    print("-" * 80)
    print("ROUTING SUMMARY TABLE")
    print("-" * 80)
    header = f"{'Scenario':<28} {'Model':<40} {'Conf.':<8} {'Cost':<12} {'Latency':<10} {'Abstain':<8} {'Escalate':<9} {'Verif.':<6}"
    print(header)
    print("-" * len(header))
    for i, (desc, dec) in enumerate(results, 1):
        plan = dec.selected_plan
        if plan and plan.plan_type == "multi_stage":
            model_label = " + ".join(sorted({sr.selected_model for sr in plan.stage_routes}))
        elif plan:
            model_label = plan.selected_model
        else:
            model_label = "NONE"
        conf = f"{plan.confidence:.3f}" if plan else "N/A"
        cost = f"${plan.expected_cost_usd:.5f}" if plan else "N/A"
        latency = f"{plan.expected_latency_ms:.0f}ms" if plan else "N/A"
        verifiers = len(plan.verifier_models) if plan else 0
        print(f"{f'{i}. {desc}':<28} {model_label:<40.40} {conf:<8} {cost:<12} {latency:<10} "
              f"{str(dec.abstain):<8} {str(dec.escalate_to_human):<9} {verifiers:<6}")
    print()
    print("=" * 80)
    print("END OF DEMONSTRATION")
    print("=" * 80)


if __name__ == "__main__":
    run()
