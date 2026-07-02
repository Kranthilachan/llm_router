"""
Optimization and Decision Making layer.

Contains:
  - Signal maps translating task family / domain into registry lookups.
  - Contextual Bayesian quality estimation (priors + task fit + domain
    expertise + runtime health).
  - Pareto frontier reduction to eliminate dominated candidates.
  - Workflow-specific utility scoring (configurable optimization profiles).
  - Confidence estimation via Thompson Sampling.

This module is pure math over plain dicts/dataclasses — no I/O, no
registry mutation (call sites always work on deepcopies), which keeps it
easy to unit test and safe to call concurrently.
"""

import math
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

from app.models.schemas import TaskFeatures

# --------------------------------------------------------------------------
# Signal maps
# --------------------------------------------------------------------------

TASK_FAMILY_SIGNAL_MAP: Dict[str, Dict[str, float]] = {
    "coding": {
        "perf:coding": 0.50, "perf:agentic_tasks": 0.15,
        "bench:swe_bench": 0.15, "bench:humaneval": 0.10,
        "perf:instruction_following": 0.10
    },
    "reasoning": {
        "perf:reasoning": 0.42, "perf:scientific_reasoning": 0.18,
        "bench:gpqa": 0.20, "bench:mmlu": 0.10,
        "perf:instruction_following": 0.10
    },
    "mathematics": {
        "perf:mathematics": 0.50, "bench:aime": 0.25,
        "perf:reasoning": 0.15, "bench:gpqa": 0.10
    },
    "chat": {
        "perf:creative_writing": 0.20, "perf:instruction_following": 0.40,
        "domain:general": 0.20, "domain:customer_support": 0.20
    },
    "vision": {
        "perf:vision_understanding": 0.50, "bench:mmmu": 0.25,
        "perf:ocr": 0.10, "perf:table_understanding": 0.15
    },
    "ocr": {
        "perf:ocr": 0.55, "bench:docvqa": 0.20,
        "perf:table_understanding": 0.15, "perf:document_qa": 0.10
    },
    "document_qa": {
        "perf:document_qa": 0.40, "perf:long_context": 0.20,
        "bench:docvqa": 0.15, "perf:summarization": 0.10,
        "perf:table_understanding": 0.15
    },
    "summarization": {
        "perf:summarization": 0.50, "perf:long_context": 0.20,
        "perf:instruction_following": 0.20, "perf:document_qa": 0.10
    },
    "translation": {
        "perf:translation": 0.70, "perf:instruction_following": 0.30
    },
    "agent": {
        "perf:agentic_tasks": 0.45, "perf:instruction_following": 0.20,
        "perf:reasoning": 0.20, "perf:coding": 0.15
    },
    "audio": {
        "perf:audio_understanding": 0.60, "perf:summarization": 0.20,
        "perf:instruction_following": 0.20
    }
}

DOMAIN_SAFETY_MAP: Dict[str, str] = {
    "medical": "medical", "legal": "legal", "finance": "finance",
    "security": "cybersecurity", "general": "general",
    "software": "general", "science": "general", "research": "general",
    "education": "general", "mathematics": "general",
    "customer_support": "general",
    "crm": "general", "hrm": "general", "project": "general", "accounts": "general"
}

DOMAIN_EXPERTISE_MAP: Dict[str, str] = {
    "medical": "medical", "legal": "legal", "finance": "finance",
    "security": "software", "software": "software", "science": "science",
    "research": "research", "education": "education", "general": "general",
    "mathematics": "mathematics", "customer_support": "customer_support",
    "crm": "customer_support", "hrm": "customer_support",
    "project": "software", "accounts": "finance"
}

WEIGHT_PROFILES = {
    "quality_first": {"quality": 0.36, "uncertainty": 0.16, "cost": 0.10, "latency": 0.08, "reliability": 0.14, "riskfit": 0.10, "runtime": 0.06},
    "budget_first": {"quality": 0.22, "uncertainty": 0.10, "cost": 0.30, "latency": 0.12, "reliability": 0.10, "riskfit": 0.08, "runtime": 0.08},
    "latency_first": {"quality": 0.20, "uncertainty": 0.10, "cost": 0.10, "latency": 0.30, "reliability": 0.10, "riskfit": 0.08, "runtime": 0.12},
    "balanced": {"quality": 0.28, "uncertainty": 0.12, "cost": 0.16, "latency": 0.12, "reliability": 0.12, "riskfit": 0.10, "runtime": 0.10},
    "high_risk": {"quality": 0.28, "uncertainty": 0.18, "cost": 0.04, "latency": 0.04, "reliability": 0.18, "riskfit": 0.18, "runtime": 0.10},
    "customer_support_summarization": {"quality": 0.22, "uncertainty": 0.10, "cost": 0.22, "latency": 0.18, "reliability": 0.10, "riskfit": 0.08, "runtime": 0.10},
    "contract_review_intake": {"quality": 0.30, "uncertainty": 0.15, "cost": 0.06, "latency": 0.06, "reliability": 0.15, "riskfit": 0.18, "runtime": 0.10},
    "coding_assistant": {"quality": 0.32, "uncertainty": 0.14, "cost": 0.10, "latency": 0.10, "reliability": 0.12, "riskfit": 0.08, "runtime": 0.14},
    "research_drafting": {"quality": 0.34, "uncertainty": 0.14, "cost": 0.08, "latency": 0.06, "reliability": 0.12, "riskfit": 0.10, "runtime": 0.16},
    "invoice_ocr_pipeline": {"quality": 0.28, "uncertainty": 0.14, "cost": 0.12, "latency": 0.08, "reliability": 0.12, "riskfit": 0.10, "runtime": 0.16},
    "multilingual_chat": {"quality": 0.24, "uncertainty": 0.10, "cost": 0.14, "latency": 0.18, "reliability": 0.10, "riskfit": 0.08, "runtime": 0.16},
    "real_time_voice_agent": {"quality": 0.20, "uncertainty": 0.10, "cost": 0.10, "latency": 0.24, "reliability": 0.12, "riskfit": 0.08, "runtime": 0.16},
    "audio_summary": {"quality": 0.26, "uncertainty": 0.12, "cost": 0.14, "latency": 0.14, "reliability": 0.12, "riskfit": 0.08, "runtime": 0.14},
    "generic_balanced": {"quality": 0.28, "uncertainty": 0.12, "cost": 0.16, "latency": 0.12, "reliability": 0.12, "riskfit": 0.10, "runtime": 0.10},
}


# --------------------------------------------------------------------------
# Signal lookup / fit functions
# --------------------------------------------------------------------------

def signal_value(model: Dict[str, Any], signal: str) -> float:
    kind, key = signal.split(":", 1)
    if kind == "perf":
        return model["performance"].get(key, 0.5)
    if kind == "bench":
        return model["benchmarks"].get(key, 0.5)
    if kind == "domain":
        return model["domains"].get(key, 0.5)
    return 0.5


def family_fit(model: Dict[str, Any], task: TaskFeatures) -> float:
    sigmap = TASK_FAMILY_SIGNAL_MAP.get(task.primary_family)
    if not sigmap:
        return model["domains"].get("general", 0.5)
    base = sum(signal_value(model, sig) * w for sig, w in sigmap.items())
    domkey = DOMAIN_EXPERTISE_MAP.get(task.domain, "general")
    domfit = model["domains"].get(domkey, 0.5)
    fit = 0.75 * base + 0.25 * domfit
    if task.expected_output == "structured_json":
        fit = 0.70 * fit + 0.30 * model["performance"].get("json_reliability", 0.5)
    if task.min_context_window > 100_000:
        fit = 0.70 * fit + 0.30 * model["performance"].get("long_context", 0.5)
    if task.requires_ocr:
        fit = 0.70 * fit + 0.30 * model["performance"].get("ocr", 0.5)
    if "spreadsheet" in task.input_formats:
        fit = 0.80 * fit + 0.20 * model["performance"].get("spreadsheet_reasoning", 0.5)
    if "audio" in task.input_formats:
        fit = 0.70 * fit + 0.30 * model["performance"].get("audio_understanding", 0.5)
    return float(min(max(fit, 0.0), 1.0))


def stage_fit(model: Dict[str, Any], stage_name: str, task: TaskFeatures) -> float:
    stage_map = {
        "domain_reasoning": lambda: model["domains"].get(DOMAIN_EXPERTISE_MAP.get(task.domain, "general"), 0.5),
        "structured_output": lambda: model["performance"].get("json_reliability", 0.5),
        "audio_understanding": lambda: model["performance"].get("audio_understanding", 0.5),
        "vision_understanding": lambda: model["performance"].get("vision_understanding", 0.5),
        "ocr": lambda: model["performance"].get("ocr", 0.5),
        "coding": lambda: model["performance"].get("coding", 0.5),
        "document_qa": lambda: model["performance"].get("document_qa", 0.5),
        "summarization": lambda: model["performance"].get("summarization", 0.5),
    }
    if stage_name in stage_map:
        return stage_map[stage_name]()
    return family_fit(model, task)


def composite_workflow_fit(model: Dict[str, Any], task: TaskFeatures) -> float:
    scores = [stage_fit(model, stage, task) for stage in task.required_stages]
    if not scores:
        return family_fit(model, task)
    return float(np.mean(scores))


def risk_support(model: Dict[str, Any], task: TaskFeatures) -> float:
    safety_key = DOMAIN_SAFETY_MAP.get(task.domain, "general")
    base = model["safety"].get(safety_key, model["safety"].get("general", 0.7))
    if task.risk_type == "regulated_advice":
        base = 0.8 * base + 0.2 * model["safety"].get("regulated_advice", base)
    if task.risk_tier == "low":
        return min(1.0, base + 0.05)
    return base


# --------------------------------------------------------------------------
# Static / runtime operational scoring
# --------------------------------------------------------------------------

def static_cost_score(model: Dict[str, Any]) -> float:
    return model["pricing"]["relative_cost_score"]


def static_latency_score(model: Dict[str, Any]) -> float:
    return model["ops_static"]["latency_score"]


def reliability_score(model: Dict[str, Any]) -> float:
    return model["ops_static"]["reliability"]


def freshness_score(model: Dict[str, Any]) -> float:
    sec = model["ops_dynamic"].get("telemetry_freshness_sec", 60)
    return max(0.0, 1.0 - min(sec / 600.0, 1.0))


def incident_penalty(status: str) -> float:
    return {"green": 0.0, "yellow": 0.08, "orange": 0.18, "red": 0.35}.get(status, 0.10)


def runtime_health_score(model: Dict[str, Any]) -> float:
    dyn = model["ops_dynamic"]
    components = [
        (0.20, max(0.0, 1.0 - min(dyn["recent_latency_ms"] / 5000.0, 1.0))),
        (0.18, max(0.0, 1.0 - min(dyn["recent_failure_rate"] / 0.15, 1.0))),
        (0.16, dyn["current_availability"]),
        (0.10, max(0.0, 1.0 - dyn["rate_limit_pressure"])),
        (0.08, max(0.0, 1.0 - dyn["queue_pressure"])),
        (0.10, max(0.0, 1.0 - incident_penalty(dyn["incident_status"]))),
        (0.08, max(0.0, 1.0 - 0.5 * dyn["budget_pressure"])),
        (0.10, freshness_score(model)),
    ]
    return float(min(max(sum(w * v for w, v in components), 0.0), 1.0))


def effective_prior(model: Dict[str, Any], family: str) -> Tuple[float, float]:
    fam = model["priors"]["task_family"].get(family, model["priors"]["global"])
    alpha, beta = float(fam["alpha"]), float(fam["beta"])
    ev = model.get("evaluation", {})
    if ev.get("samples", 0) > 0:
        alpha += ev.get("wins", 0)
        beta += ev.get("losses", 0)
    return alpha, beta


def estimate_request_cost_usd(model: Dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    return ((input_tokens / 1_000_000) * model["pricing"]["input_cost"] +
            (output_tokens / 1_000_000) * model["pricing"]["output_cost"])


def estimate_request_latency_ms(model: Dict[str, Any], task: TaskFeatures, n_stages: int = 1) -> float:
    base = model["ops_dynamic"]["recent_latency_ms"]
    complexity_mult = {"low": 1.0, "medium": 1.25, "high": 1.60}.get(task.complexity, 1.0)
    modality_mult = 1.0
    if "audio" in task.input_formats or "video" in task.input_formats:
        modality_mult += 0.25
    if "pdf" in task.input_formats and task.requires_ocr:
        modality_mult += 0.20
    return float(base * complexity_mult * modality_mult * n_stages)


# --------------------------------------------------------------------------
# Bayesian contextual quality
# --------------------------------------------------------------------------

def beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def beta_variance(alpha: float, beta: float) -> float:
    return (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))


def convex_combine(values: List[float], weights: List[float]) -> float:
    total = sum(weights)
    if total == 0:
        return float(np.mean(values))
    return sum(v * w for v, w in zip(values, weights)) / total


def attach_contextual_quality(
    models: List[Dict[str, Any]],
    task: TaskFeatures,
) -> List[Dict[str, Any]]:
    enriched = []
    for model in models:
        model = deepcopy(model)
        g = model["priors"]["global"]
        global_mean = beta_mean(g["alpha"], g["beta"])
        falpha, fbeta = effective_prior(model, task.primary_family)
        fam_mean = beta_mean(falpha, fbeta)
        fam_var = beta_variance(falpha, fbeta)
        fam_fit = family_fit(model, task)
        wf_fit = composite_workflow_fit(model, task)
        dom_fit = model["domains"].get(DOMAIN_EXPERTISE_MAP.get(task.domain, "general"), 0.5)
        rt_fit = runtime_health_score(model)
        fr_fit = freshness_score(model)
        contextual_mean = convex_combine(
            [global_mean, fam_mean, fam_fit, wf_fit, dom_fit, fr_fit],
            [0.12, 0.22, 0.18, 0.23, 0.15, 0.10]
        )
        runtime_adjusted_mean = 0.76 * contextual_mean + 0.24 * rt_fit
        model["q"] = {
            "global_mean": global_mean,
            "family_mean": fam_mean,
            "family_variance": fam_var,
            "family_fit": fam_fit,
            "workflow_fit": wf_fit,
            "domain_fit": dom_fit,
            "runtime_fit": rt_fit,
            "freshness_fit": fr_fit,
            "contextual_mean": contextual_mean,
            "runtime_adjusted_mean": runtime_adjusted_mean,
            "uncertainty": math.sqrt(max(fam_var, 1e-9)),
            "alpha": falpha,
            "beta": fbeta,
        }
        enriched.append(model)
    return enriched


# --------------------------------------------------------------------------
# Pareto frontier reduction
# --------------------------------------------------------------------------

def pareto_vector(model: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        model["q"]["runtime_adjusted_mean"],
        -static_cost_score(model),
        -static_latency_score(model),
        reliability_score(model),
        runtime_health_score(model),
    )


def dominates_with_margin(m1: Dict[str, Any], m2: Dict[str, Any], eps: float = 0.02) -> bool:
    v1 = pareto_vector(m1)
    v2 = pareto_vector(m2)
    ge_all = all(a >= b - eps for a, b in zip(v1, v2))
    gt_any = any(a > b + eps for a, b in zip(v1, v2))
    return ge_all and gt_any


def pareto_frontier(models: List[Dict[str, Any]], eps: float = 0.02) -> List[Dict[str, Any]]:
    frontier = []
    for i, model in enumerate(models):
        dominated = any(
            dominates_with_margin(models[j], model, eps=eps)
            for j in range(len(models)) if j != i
        )
        if not dominated:
            frontier.append(model)
    return frontier


# --------------------------------------------------------------------------
# Utility scoring
# --------------------------------------------------------------------------

def minmax_normalize(values: List[float], invert: bool = False) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [0.5] * len(values)
    base = [(v - vmin) / (vmax - vmin) for v in values]
    return [1 - x for x in base] if invert else base


def choose_effective_profile(task: TaskFeatures, requested: str) -> str:
    if task.risk_tier == "high":
        return "high_risk"
    if task.workflow_profile in WEIGHT_PROFILES:
        return task.workflow_profile
    return requested if requested in WEIGHT_PROFILES else "balanced"


def compute_utilities(
    models: List[Dict[str, Any]],
    task: TaskFeatures,
    profile_name: str = "balanced",
) -> List[Dict[str, Any]]:
    profile_name = choose_effective_profile(task, profile_name)
    w = WEIGHT_PROFILES[profile_name]

    qvals = [m["q"]["runtime_adjusted_mean"] for m in models]
    uvals = [m["q"]["uncertainty"] for m in models]
    cvals = [static_cost_score(m) for m in models]
    lvals = [static_latency_score(m) for m in models]
    rvals = [reliability_score(m) for m in models]
    rfvals = [risk_support(m, task) for m in models]
    rtvals = [runtime_health_score(m) for m in models]

    qn = minmax_normalize(qvals)
    un = minmax_normalize(uvals)
    cn = minmax_normalize(cvals)
    ln = minmax_normalize(lvals)
    rn = minmax_normalize(rvals)
    rfn = minmax_normalize(rfvals)
    rtn = minmax_normalize(rtvals)

    out = []
    for model, q, u, c, l, r, rf, rt in zip(models, qn, un, cn, ln, rn, rfn, rtn):
        m = deepcopy(model)
        utility = (
            w["quality"] * q
            - w["uncertainty"] * u
            - w["cost"] * c
            - w["latency"] * l
            + w["reliability"] * r
            + w["riskfit"] * rf
            + w["runtime"] * rt
        )
        if task.ambiguity_score > 0.5:
            utility -= 0.03
        if task.decomposition_needed and len(task.required_stages) > 2:
            utility -= 0.01

        rc = task.request_constraints
        est_latency = estimate_request_latency_ms(model, task)
        est_cost = estimate_request_cost_usd(model, task.estimated_tokens, task.estimated_output_tokens)
        sla_violation = False
        if rc.max_latency_ms is not None and est_latency > rc.max_latency_ms:
            utility -= 0.15
            sla_violation = True
        if rc.max_cost_usd is not None and est_cost > rc.max_cost_usd:
            utility -= 0.15
            sla_violation = True

        m["u"] = {
            "profile_used": profile_name,
            "expected_utility": utility,
            "n_quality": q, "n_uncertainty": u, "n_cost": c,
            "n_latency": l, "n_reliability": r, "n_riskfit": rf, "n_runtime": rt,
            "est_latency_ms": est_latency,
            "est_cost_usd": est_cost,
            "sla_violation": sla_violation,
        }
        out.append(m)
    return sorted(out, key=lambda x: x["u"]["expected_utility"], reverse=True)


# --------------------------------------------------------------------------
# Confidence estimation via Thompson Sampling
# --------------------------------------------------------------------------

def bounded_beta_sample(mean: float, variance: float) -> float:
    max_var = max(mean * (1 - mean) - 1e-6, 1e-6)
    variance = max(min(variance, max_var), 1e-6)
    common = (mean * (1 - mean) / variance) - 1
    alpha = max(mean * common, 1.0)
    beta_param = max((1 - mean) * common, 1.0)
    return float(np.random.beta(alpha, beta_param))


def estimate_confidence(
    models: List[Dict[str, Any]],
    task: TaskFeatures,
    profile_name: str = "balanced",
    n_sim: int = 1500,
) -> Dict[str, Any]:
    profile_name = choose_effective_profile(task, profile_name)
    w = WEIGHT_PROFILES[profile_name]
    wins = {m["name"]: 0 for m in models}
    for _ in range(n_sim):
        sampled_utilities = []
        for m in models:
            sq = bounded_beta_sample(m["q"]["runtime_adjusted_mean"], max(m["q"]["family_variance"], 1e-4))
            su = (
                w["quality"] * sq
                - w["cost"] * static_cost_score(m)
                + w["reliability"] * reliability_score(m)
                + w["runtime"] * runtime_health_score(m)
            )
            sampled_utilities.append((m["name"], su))
        winner = max(sampled_utilities, key=lambda x: x[1])[0]
        wins[winner] += 1
    win_probs = {k: v / n_sim for k, v in wins.items()}
    ordered = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    top_name, top_prob = ordered[0]
    second_prob = ordered[1][1] if len(ordered) > 1 else 0.0
    return {
        "win_probabilities": win_probs,
        "top_model": top_name,
        "top_confidence": top_prob,
        "margin": top_prob - second_prob,
        "ordered": ordered,
        "n_simulations": n_sim,
        "calibration_note": "Calibrate against held-out labeled routing outcomes for production use.",
    }
