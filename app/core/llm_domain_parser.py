"""
LLM tiebreaker for domain classification on ambiguous text prompts.

Wired into route() via the `llm_parser=` seam. Runs after the heuristic +
embedding classifier. If the embedding classifier's cosine similarity is
below LEARNED_DOMAIN_CONFIDENCE_THRESHOLD AND the prompt is text-only AND
the heuristic didn't already flag a high-risk regulated-advice path, we
call gpt-4o-mini with a structured-output schema to reclassify.

Design guarantees:
- Never overrides regulated-advice keyword paths (medical/legal/finance/
  security). Those are safety-critical and set by keywords first.
- Skipped for multimodal prompts (image/pdf/audio/video) — existing
  modality-based branches in semantic_parser.py already route those
  correctly.
- Graceful failure — on any LLM error, fall back to the heuristic result
  so /route still succeeds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from app.core.domain_classifier import (
    LEARNED_DOMAIN_CONFIDENCE_THRESHOLD,
    classify_domain,
)
from app.core.semantic_parser import (
    fallback_structured_semantic_parse,
    validate_structured_parse,
)
from app.models.schemas import ArtifactProfile, StructuredSemanticParse

logger = logging.getLogger(__name__)


# ---- JSON schema for OpenAI structured outputs ----
# Excludes medical/legal/finance/security from allowed domains — those are
# keyword-only paths. Excludes risk_tier="high" — only keywords may set high.

_ALLOWED_FAMILIES = [
    "chat", "coding", "reasoning", "mathematics", "vision", "ocr",
    "document_qa", "audio", "agent", "translation", "summarization",
]
_ALLOWED_DOMAINS = [
    "crm", "hrm", "project", "accounts", "customer_support",
    "software", "science", "research", "education", "mathematics", "general",
]
_ALLOWED_RISK_TIERS = ["low", "medium"]
_ALLOWED_RISK_TYPES = ["standard", "operational", "analytical", "pii_sensitive"]
_ALLOWED_OUTPUTS = ["free_text", "structured_json", "code"]
_ALLOWED_ACTIONABILITY = ["advisory", "high"]

_JSON_SCHEMA = {
    "name": "atlas_semantic_parse",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "primary_family", "domain", "risk_tier", "risk_type",
            "expected_output", "ambiguity_score", "actionability",
            "document_type", "parser_confidence", "reason_summary",
        ],
        "properties": {
            "primary_family": {"type": "string", "enum": _ALLOWED_FAMILIES},
            "domain": {"type": "string", "enum": _ALLOWED_DOMAINS},
            "risk_tier": {"type": "string", "enum": _ALLOWED_RISK_TIERS},
            "risk_type": {"type": "string", "enum": _ALLOWED_RISK_TYPES},
            "expected_output": {"type": "string", "enum": _ALLOWED_OUTPUTS},
            "ambiguity_score": {"type": "number", "minimum": 0, "maximum": 1},
            "actionability": {"type": "string", "enum": _ALLOWED_ACTIONABILITY},
            "document_type": {"type": "string"},
            "parser_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_summary": {"type": "string"},
        },
    },
}

_SYSTEM_PROMPT = (
    "You classify user prompts for an LLM routing service. Emit ONLY the JSON "
    "schema fields. Rules:\n"
    "- domain must be one of: " + ", ".join(_ALLOWED_DOMAINS) + ".\n"
    "- Never emit medical/legal/finance/security as domain — those are handled "
    "by upstream keyword rules. If the prompt is regulated advice, pick the "
    "closest allowed domain (e.g. general).\n"
    "- risk_tier is low or medium only. Never emit high.\n"
    "- primary_family is the task modality: coding for code, mathematics for "
    "proofs/derivations, vision/ocr/document_qa for file-driven tasks, chat "
    "otherwise.\n"
    "- Set parser_confidence to your best estimate (0.5-1.0).\n"
    "- reason_summary: one short sentence stating which domain and why."
)


def _call_openai_parser(
    prompt: str,
    input_formats: Optional[List[str]] = None,
    estimated_tokens: int = 2000,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> StructuredSemanticParse:
    """Call gpt-4o-mini with structured outputs; return a StructuredSemanticParse."""
    from openai import OpenAI
    from app.config import ATLAS_LLM_PARSER_MODEL

    client = OpenAI()
    completion = client.chat.completions.create(
        model=ATLAS_LLM_PARSER_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": _JSON_SCHEMA},
        temperature=0,
    )
    payload = json.loads(completion.choices[0].message.content)

    parsed = StructuredSemanticParse(
        primary_family=payload["primary_family"],
        secondary_families=[],
        required_stages=[payload["primary_family"]],
        workflow_graph=[{"stage_id": 1, "stage_name": payload["primary_family"], "depends_on": []}],
        domain=payload["domain"],
        risk_tier=payload["risk_tier"],
        risk_type=payload["risk_type"],
        expected_output=payload["expected_output"],
        ambiguity_score=payload["ambiguity_score"],
        actionability=payload["actionability"],
        document_type=payload["document_type"],
        decomposition_needed=False,
        needs_verification=False,
        parser_confidence=payload["parser_confidence"],
        reason_summary=f"LLM tiebreaker: {payload['reason_summary']}",
    )
    return validate_structured_parse(parsed)


def hybrid_tiebreaker_parser(
    prompt: str,
    input_formats: Optional[List[str]] = None,
    estimated_tokens: int = 2000,
    artifacts: Optional[List[Any]] = None,
) -> StructuredSemanticParse:
    """
    Run the heuristic + embedding classifier first. Escalate to the LLM only
    when the prompt is text-only, the embedding classifier is uncertain, and
    the heuristic didn't already set a regulated-advice high-risk path.
    """
    # Coerce artifacts to ArtifactProfile shape expected by the fallback.
    # The seam at semantic_parser.py:227 passes them as dicts (via asdict);
    # the fallback function expects a list of ArtifactProfile-like objects.
    fallback_artifacts: List[ArtifactProfile] = []
    for a in artifacts or []:
        if isinstance(a, ArtifactProfile):
            fallback_artifacts.append(a)
        elif isinstance(a, dict):
            fallback_artifacts.append(ArtifactProfile(**a))

    heuristic = fallback_structured_semantic_parse(
        prompt, input_formats or ["text"], estimated_tokens, fallback_artifacts
    )

    is_text_only = set(input_formats or ["text"]) <= {"text"}
    _, embed_sim = classify_domain(prompt)
    is_uncertain = embed_sim < LEARNED_DOMAIN_CONFIDENCE_THRESHOLD
    is_regulated = heuristic.risk_tier == "high"

    if is_text_only and is_uncertain and not is_regulated:
        try:
            return _call_openai_parser(
                prompt=prompt,
                input_formats=input_formats,
                estimated_tokens=estimated_tokens,
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.warning(
                "LLM tiebreaker failed (%s: %s); using heuristic result.",
                type(exc).__name__, exc,
            )

    return heuristic
