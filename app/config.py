"""
ATLAS Router — system constants and versioning.

Every production routing system must be reproducible under audit. Each
independently-evolving component carries its own version stamp so that a
routing decision questioned weeks later can be reconstructed exactly.

Version axes:
    ROUTER_VERSION      -> orchestration logic (the route() pipeline)
    PARSER_VERSION       -> semantic interpretation layer
    POLICY_VERSION        -> governance rules
    SCORING_VERSION        -> utility / quality math
    REGISTRY_VERSION         -> canonical model registry snapshot
    CALIBRATION_VERSION       -> Bayesian prior calibration snapshot
    TELEMETRY_SNAPSHOT_VERSION -> runtime telemetry snapshot
"""

import os
from typing import List, Optional

from dotenv import load_dotenv

# Populate os.environ from the project's .env file (repo root) so that
# ATLAS_ALLOWED_PROVIDERS / other config vars edited there are picked up
# without needing to `export` them in the shell. Values already set in
# the process environment win over the file, matching dotenv's default.
load_dotenv()


def _parse_provider_env(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    providers = [p.strip() for p in raw.split(",") if p.strip()]
    return providers or None


# Default provider allowlist applied to every /route request when the caller
# doesn't set its own `request_constraints.allowed_providers`. Callers can
# still override per-request. Set to e.g. "OpenAI" or "Anthropic" (or
# "OpenAI,Anthropic") when you only hold API keys for those providers — the
# router then only selects models (primary + fallbacks + verifiers) from that
# subset. Leave unset for the full multi-provider behavior.
DEFAULT_ALLOWED_PROVIDERS: Optional[List[str]] = _parse_provider_env(
    os.getenv("ATLAS_ALLOWED_PROVIDERS")
)


ROUTER_VERSION = "atlas-router-v1.0.0"
PARSER_VERSION = "parser-v1.0.0"
POLICY_VERSION = "policy-v1.0.0"
SCORING_VERSION = "scoring-v1.0.0"
CALIBRATION_VERSION = "calibration-v1.0.0"
REGISTRY_VERSION = "registry-v1.0.0-2026-06-snapshot"
TELEMETRY_SNAPSHOT_VERSION = "telemetry-snap-2026-06-27"

SUPPORTED_FORMATS = [
    "text", "image", "pdf", "audio", "video", "spreadsheet", "presentation"
]

# Escalation ladder: below ABSTAIN the router refuses to answer; below
# ESCALATE it answers but flags for human review; above HIGH it proceeds
# autonomously even on high-risk tasks. All three can be overridden per
# deployment via env vars (defaults kept for backward compat).
#
# The default ABSTAIN threshold (0.40) assumes a multi-provider registry
# where the top model naturally wins a large share of Thompson-Sampling
# simulations. Narrower candidate pools (e.g. single-provider setups via
# ATLAS_ALLOWED_PROVIDERS) produce lower peak confidences because models
# within one family score similarly — set ATLAS_ABSTAIN_THRESHOLD=0.30
# (or lower) to keep the router from abstaining on those.
def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


CONFIDENCE_ABSTAIN_THRESHOLD = _float_env("ATLAS_ABSTAIN_THRESHOLD", 0.40)
CONFIDENCE_ESCALATE_THRESHOLD = _float_env("ATLAS_ESCALATE_THRESHOLD", 0.55)
CONFIDENCE_HIGH_THRESHOLD = _float_env("ATLAS_HIGH_THRESHOLD", 0.75)

# Embedding model for the domain classifier (loaded once at import time
# by app/core/domain_classifier.py). Any sentence-transformers-compatible
# model works; default is a 384-dim MiniLM (~90MB, English-focused).
ATLAS_EMBEDDING_MODEL = os.getenv("ATLAS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# LLM tiebreaker: model used when the embedding classifier's cosine
# similarity is below LEARNED_DOMAIN_CONFIDENCE_THRESHOLD on a text-only
# prompt. Requires OPENAI_API_KEY. Set to "" (empty) to disable entirely.
ATLAS_LLM_PARSER_MODEL = os.getenv("ATLAS_LLM_PARSER_MODEL", "gpt-4o-mini")
LLM_PARSER_ENABLED = bool(os.getenv("OPENAI_API_KEY")) and bool(ATLAS_LLM_PARSER_MODEL)
