"""
Canonical Model Registry builder.

Turns the raw provider -> model-name catalog (catalog.py) into fully-shaped
registry entries: modalities, capabilities, context window, pricing,
static/dynamic ops telemetry, Bayesian priors, performance/benchmark
signals, domain expertise, safety scores, and verifier fitness.

Every numeric value here is a *reasonable synthetic default*, generated
deterministically (seeded per model name) so registry snapshots are
reproducible across restarts. In production this module would be replaced
by (or hydrated from) a real benchmark/telemetry pipeline; the shape of
each entry is what the rest of the router depends on, not these exact
numbers. Swap this out for a database-backed or config-driven registry
without touching feasibility/policy/scoring code.
"""

import hashlib
from typing import Any, Dict, List

from app.models.catalog import MODELS

TIER_ECONOMY_HINTS = [
    "mini", "small", "lite", "micro", "haiku", "flash", "nano", "1b", "3b",
    "distill", "instant", "7b", "mixtral-8x7b",
]
TIER_FRONTIER_HINTS = [
    "opus", "ultra", "pro", "premier", "large", "plus", "gpt-5.5", "gpt-5.4",
    "o3", "o4", "grok-4", "405b", "maverick", "v4-pro", "command-a",
]

OPEN_WEIGHT_PROVIDERS = {"Meta", "DeepSeek", "Alibaba (Qwen)", "Mistral AI", "01.AI", "NVIDIA", "IBM"}
OPEN_WEIGHT_NAME_HINTS = ["llama", "qwen", "mixtral", "mistral-7b", "gemma", "phi-3", "phi-4", "granite", "yi-", "glm-4"]

VISION_HINTS = ["vision", "4o", "gemini", "grok-4", "claude", "gpt-5", "pro", "nova", "phi-4-multimodal"]
AUDIO_HINTS = ["gemini", "gpt-5.5", "gpt-4o", "nova-premier", "grok-4"]

PERF_KEYS = [
    "coding", "agentic_tasks", "reasoning", "scientific_reasoning", "mathematics",
    "creative_writing", "instruction_following", "vision_understanding", "ocr",
    "table_understanding", "document_qa", "long_context", "summarization",
    "translation", "audio_understanding", "json_reliability",
    "spreadsheet_reasoning",
]
BENCH_KEYS = ["swe_bench", "humaneval", "gpqa", "mmlu", "aime", "mmmu", "docvqa"]
DOMAIN_KEYS = ["general", "legal", "finance", "medical", "software", "science",
               "research", "education", "mathematics", "customer_support"]
SAFETY_KEYS = ["general", "medical", "legal", "finance", "cybersecurity", "regulated_advice"]
VERIFIER_KEYS = ["safety_review", "factuality_review", "schema_validation",
                  "citation_review", "consistency_review"]


def _seeded_rng(name: str, salt: str = "") -> "_Deterministic":
    seed_bytes = hashlib.sha256(f"{name}:{salt}".encode()).digest()
    return _Deterministic(int.from_bytes(seed_bytes[:8], "big"))


class _Deterministic:
    """Tiny deterministic PRNG (xorshift64) so registry values are stable
    across process restarts without depending on global numpy RNG state."""

    def __init__(self, seed: int):
        self.state = seed or 0x9E3779B97F4A7C15

    def _next(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7)
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return self.state

    def uniform(self, lo: float, hi: float) -> float:
        frac = (self._next() % 10_000_000) / 10_000_000
        return lo + frac * (hi - lo)


def _classify_tier(name: str) -> str:
    low = name.lower()
    if any(h in low for h in TIER_ECONOMY_HINTS):
        return "Economy"
    if any(h in low for h in TIER_FRONTIER_HINTS):
        return "Frontier"
    return "Mid"


def _is_open_weight(provider: str, name: str) -> bool:
    if provider in OPEN_WEIGHT_PROVIDERS:
        return True
    return any(h in name.lower() for h in OPEN_WEIGHT_NAME_HINTS)


def _base_quality_for_tier(tier: str) -> float:
    return {"Frontier": 0.86, "Mid": 0.74, "Economy": 0.62}[tier]


def _pricing_for_tier(tier: str, rng: _Deterministic) -> Dict[str, float]:
    if tier == "Frontier":
        input_cost = rng.uniform(3.0, 15.0)
        output_cost = input_cost * rng.uniform(3.5, 6.0)
        relative_cost_score = rng.uniform(0.65, 1.0)
    elif tier == "Mid":
        input_cost = rng.uniform(0.5, 3.0)
        output_cost = input_cost * rng.uniform(3.0, 5.0)
        relative_cost_score = rng.uniform(0.30, 0.65)
    else:
        input_cost = rng.uniform(0.03, 0.5)
        output_cost = input_cost * rng.uniform(2.5, 4.5)
        relative_cost_score = rng.uniform(0.05, 0.30)
    return {
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "relative_cost_score": round(relative_cost_score, 3),
    }


def _context_window_for_tier(tier: str, rng: _Deterministic) -> int:
    if tier == "Frontier":
        return int(rng.uniform(200_000, 1_000_000))
    if tier == "Mid":
        return int(rng.uniform(64_000, 256_000))
    return int(rng.uniform(8_000, 64_000))


def _build_performance(name: str, provider: str, tier: str, rng: _Deterministic) -> Dict[str, float]:
    base = _base_quality_for_tier(tier)
    low = name.lower()
    perf = {}
    for key in PERF_KEYS:
        jitter = rng.uniform(-0.12, 0.12)
        perf[key] = round(min(max(base + jitter, 0.05), 0.99), 3)
    # Specialization boosts by name/provider hints.
    if any(h in low for h in ["coder", "codestral", "code"]):
        perf["coding"] = round(min(perf["coding"] + 0.15, 0.99), 3)
    if provider == "Anthropic" or "claude" in low:
        perf["instruction_following"] = round(min(perf["instruction_following"] + 0.08, 0.99), 3)
        perf["long_context"] = round(min(perf["long_context"] + 0.06, 0.99), 3)
    if provider == "OpenAI" and ("o3" in low or "o4" in low or "thinking" in low):
        perf["reasoning"] = round(min(perf["reasoning"] + 0.15, 0.99), 3)
        perf["mathematics"] = round(min(perf["mathematics"] + 0.12, 0.99), 3)
    if "deepseek-r1" in low or provider == "DeepSeek":
        perf["mathematics"] = round(min(perf["mathematics"] + 0.10, 0.99), 3)
        perf["reasoning"] = round(min(perf["reasoning"] + 0.10, 0.99), 3)
    if any(h in low for h in VISION_HINTS):
        perf["vision_understanding"] = round(min(perf["vision_understanding"] + 0.10, 0.99), 3)
        perf["ocr"] = round(min(perf["ocr"] + 0.08, 0.99), 3)
    if any(h in low for h in AUDIO_HINTS):
        perf["audio_understanding"] = round(min(perf["audio_understanding"] + 0.12, 0.99), 3)
    if tier == "Frontier":
        perf["json_reliability"] = round(min(perf["json_reliability"] + 0.08, 0.99), 3)
    return perf


def _build_benchmarks(base_perf: Dict[str, float], rng: _Deterministic) -> Dict[str, float]:
    mapping = {
        "swe_bench": "coding", "humaneval": "coding", "gpqa": "reasoning",
        "mmlu": "reasoning", "aime": "mathematics", "mmmu": "vision_understanding",
        "docvqa": "ocr",
    }
    out = {}
    for bench, perf_key in mapping.items():
        jitter = rng.uniform(-0.05, 0.05)
        out[bench] = round(min(max(base_perf.get(perf_key, 0.6) + jitter, 0.05), 0.99), 3)
    return out


def _build_domains(provider: str, name: str, tier: str, rng: _Deterministic) -> Dict[str, float]:
    base = _base_quality_for_tier(tier)
    domains = {k: round(min(max(base + rng.uniform(-0.15, 0.10), 0.05), 0.95), 3) for k in DOMAIN_KEYS}
    domains["general"] = round(min(base + 0.05, 0.97), 3)
    if tier == "Frontier":
        for k in ("legal", "medical", "finance"):
            domains[k] = round(min(domains[k] + 0.08, 0.95), 3)
    return domains


def _build_safety(tier: str, rng: _Deterministic) -> Dict[str, float]:
    base = {"Frontier": 0.90, "Mid": 0.80, "Economy": 0.72}[tier]
    safety = {k: round(min(max(base + rng.uniform(-0.08, 0.06), 0.4), 0.98) , 3) for k in SAFETY_KEYS}
    return safety


def _build_verifier_fit(tier: str, rng: _Deterministic) -> Dict[str, float]:
    base = {"Frontier": 0.85, "Mid": 0.72, "Economy": 0.55}[tier]
    return {k: round(min(max(base + rng.uniform(-0.10, 0.08), 0.2), 0.97), 3) for k in VERIFIER_KEYS}


def _build_ops_static(tier: str, rng: _Deterministic) -> Dict[str, float]:
    latency_score = {"Frontier": rng.uniform(0.45, 0.70),
                       "Mid": rng.uniform(0.60, 0.85),
                       "Economy": rng.uniform(0.80, 0.97)}[tier]
    reliability = round(rng.uniform(0.90, 0.995), 4)
    return {"latency_score": round(latency_score, 3), "reliability": reliability}


def _build_ops_dynamic(tier: str, rng: _Deterministic) -> Dict[str, Any]:
    base_latency_ms = {"Frontier": rng.uniform(1800, 4500),
                         "Mid": rng.uniform(700, 2000),
                         "Economy": rng.uniform(200, 900)}[tier]
    incident_roll = rng.uniform(0, 1)
    incident_status = "green" if incident_roll < 0.90 else ("yellow" if incident_roll < 0.97 else "orange")
    return {
        "recent_latency_ms": round(base_latency_ms, 1),
        "recent_failure_rate": round(rng.uniform(0.001, 0.04), 4),
        "current_availability": round(rng.uniform(0.97, 0.999), 4),
        "rate_limit_pressure": round(rng.uniform(0.0, 0.35), 3),
        "queue_pressure": round(rng.uniform(0.0, 0.30), 3),
        "incident_status": incident_status,
        "budget_pressure": round(rng.uniform(0.0, 0.30), 3),
        "telemetry_freshness_sec": int(rng.uniform(5, 180)),
    }


def _build_priors(name: str, rng: _Deterministic) -> Dict[str, Any]:
    g_alpha = rng.uniform(8, 20)
    g_beta = rng.uniform(2, 10)
    task_family = {}
    for fam in ["coding", "reasoning", "mathematics", "chat", "vision", "ocr",
                "document_qa", "summarization", "translation", "agent", "audio"]:
        task_family[fam] = {
            "alpha": round(rng.uniform(4, 16), 2),
            "beta": round(rng.uniform(1, 8), 2),
        }
    return {
        "global": {"alpha": round(g_alpha, 2), "beta": round(g_beta, 2)},
        "task_family": task_family,
    }


def _build_evaluation() -> Dict[str, Any]:
    return {
        "samples": 0, "wins": 0, "losses": 0,
        "quality_sum": 0.0, "latency_sum_ms": 0.0, "cost_sum": 0.0,
        "user_acceptance_sum": 0.0, "safety_flags": 0,
        "last_updated": None,
    }


def _build_modalities(name: str, tier: str) -> Dict[str, bool]:
    low = name.lower()
    supports_vision = any(h in low for h in VISION_HINTS) or tier == "Frontier"
    supports_audio = any(h in low for h in AUDIO_HINTS)
    supports_video = supports_audio and tier in ("Frontier", "Mid")
    return {
        "text": True,
        "image": supports_vision,
        "pdf": True,  # all models accept PDF as extracted text at minimum
        "audio": supports_audio,
        "video": supports_video,
        "spreadsheet": True,
        "presentation": True,
    }


def _build_capabilities(name: str, tier: str) -> Dict[str, bool]:
    low = name.lower()
    return {
        "json_mode": tier in ("Frontier", "Mid") or "instruct" in low,
        "function_calling": tier in ("Frontier", "Mid"),
        "web_search": tier == "Frontier" or "gpt" in low or "gemini" in low or "grok" in low,
        "ocr": _build_modalities(name, tier)["image"] or _build_modalities(name, tier)["pdf"],
        "citation_support": tier in ("Frontier", "Mid"),
    }


def _routing_meta(domain_high_risk_penalty: bool) -> Dict[str, Any]:
    return {"avoid_for": ["high_risk"] if domain_high_risk_penalty else []}


def build_registry() -> List[Dict[str, Any]]:
    registry: List[Dict[str, Any]] = []
    for provider, names in MODELS.items():
        for name in names:
            rng = _seeded_rng(name, salt="atlas-registry-v1")
            tier = _classify_tier(name)
            perf = _build_performance(name, provider, tier, rng)
            entry: Dict[str, Any] = {
                "name": name,
                "provider": provider,
                "tier": tier,
                "status": "active",
                "api_available": True,
                "open_weight": _is_open_weight(provider, name),
                "allowed_regions": ["global"],
                "modalities": _build_modalities(name, tier),
                "capabilities": _build_capabilities(name, tier),
                "context": {"window": _context_window_for_tier(tier, rng)},
                "pricing": _pricing_for_tier(tier, rng),
                "ops_static": _build_ops_static(tier, rng),
                "ops_dynamic": _build_ops_dynamic(tier, rng),
                "priors": _build_priors(name, rng),
                "evaluation": _build_evaluation(),
                "performance": perf,
                "benchmarks": _build_benchmarks(perf, rng),
                "domains": _build_domains(provider, name, tier, rng),
                "safety": _build_safety(tier, rng),
                "routing": _routing_meta(tier == "Economy"),
                "verifier_fit": _build_verifier_fit(tier, rng),
            }
            registry.append(entry)
    return registry


# Built once at import time; treated as the live canonical registry for the
# process. In production, swap this for a periodically-refreshed snapshot
# (e.g. loaded from a config service / database) — see app/api/routes.py
# for where a reload hook would live.
MODEL_REGISTRY: List[Dict[str, Any]] = build_registry()
