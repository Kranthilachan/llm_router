"""
Core data contracts used throughout the ATLAS routing pipeline.

These are plain dataclasses (not pydantic) because they flow through pure
scoring/optimization code as well as the FastAPI layer. The API layer
(app/api/routes.py) defines separate pydantic request/response models and
converts to/from these dataclasses at the boundary.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------------
# Artifact inspection
# --------------------------------------------------------------------------

@dataclass
class ArtifactProfile:
    format: str
    page_count: Optional[int] = None
    text_density: Optional[float] = None
    scan_likelihood: Optional[float] = None
    handwriting_likelihood: Optional[float] = None
    table_density: Optional[float] = None
    chart_density: Optional[float] = None
    detected_language: Optional[str] = None
    audio_duration_sec: Optional[int] = None
    audio_quality: Optional[float] = None
    video_duration_sec: Optional[int] = None
    spreadsheet_complexity: Optional[float] = None
    presentation_complexity: Optional[float] = None
    # ---- file-driven pipeline fields ----
    source_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    extracted_text_preview: Optional[str] = None
    inferred_topic: Optional[str] = None
    extraction_method: str = "heuristic"


# --------------------------------------------------------------------------
# Request-time constraints supplied by the caller / tenant
# --------------------------------------------------------------------------

@dataclass
class RequestConstraints:
    allowed_providers: Optional[List[str]] = None
    disallowed_providers: Optional[List[str]] = None
    allowed_tiers: Optional[List[str]] = None
    no_open_weight: bool = False
    required_region: Optional[str] = None
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    mandatory_verifier: bool = False
    no_web_access: bool = False
    must_use_single_model: bool = False
    min_confidence: float = 0.0


@dataclass
class TenantContext:
    tenant_id: Optional[str] = None
    tenant_name: Optional[str] = None
    allowed_models: Optional[List[str]] = None
    budget_remaining_usd: Optional[float] = None
    # e.g. {"frontier_only": True, "allowed_providers": [...], "mandatory_verifier": True}
    policy_overlay: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------
# Semantic parsing output
# --------------------------------------------------------------------------

@dataclass
class StructuredSemanticParse:
    primary_family: str
    secondary_families: List[str] = field(default_factory=list)
    required_stages: List[str] = field(default_factory=list)
    workflow_graph: List[Dict[str, Any]] = field(default_factory=list)
    domain: str = "general"
    risk_tier: str = "low"
    risk_type: str = "standard"
    expected_output: str = "free_text"
    ambiguity_score: float = 0.15
    actionability: str = "advisory"
    document_type: str = "generic"
    decomposition_needed: bool = False
    needs_verification: bool = False
    parser_confidence: float = 0.78
    reason_summary: str = ""


# --------------------------------------------------------------------------
# Unified task representation — the single object the rest of the
# pipeline (feasibility, policy, scoring, planning) consumes.
# --------------------------------------------------------------------------

@dataclass
class TaskFeatures:
    raw_prompt: str
    input_formats: List[str]
    estimated_tokens: int
    estimated_output_tokens: int
    artifacts: List[ArtifactProfile]

    primary_family: str
    secondary_families: List[str]
    required_stages: List[str]
    workflow_graph: List[Dict[str, Any]]

    complexity: str
    domain: str
    risk_tier: str
    risk_type: str

    required_formats: List[str]
    requires_json: bool
    requires_function_calling: bool
    requires_web_search: bool
    requires_ocr: bool
    requires_citations: bool
    requires_verifier: bool
    min_context_window: int

    expected_output: str
    ambiguity_score: float
    safety_sensitive: bool
    actionability: str
    document_type: str
    decomposition_needed: bool
    workflow_profile: str
    parser_confidence: float

    conflict_flags: List[str]
    extracted_text_summary: Optional[str]
    detected_languages: List[str]
    total_file_size_bytes: int

    request_constraints: RequestConstraints
    tenant_context: TenantContext


# --------------------------------------------------------------------------
# Governance
# --------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    allowed: bool
    must_escalate: bool
    must_abstain: bool
    deny_reason: Optional[str] = None
    restricted_to_tiers: List[str] = field(default_factory=list)
    restricted_to_models: List[str] = field(default_factory=list)
    restricted_to_providers: List[str] = field(default_factory=list)
    require_verifier_types: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Execution planning
# --------------------------------------------------------------------------

@dataclass
class StageRoute:
    stage_id: int
    stage_name: str
    selected_model: str
    fallback_models: List[str]
    verifier_models: List[str]
    stage_confidence: float
    expected_latency_ms: float
    expected_cost_usd: float
    explanation: str


@dataclass
class ExecutionPlan:
    plan_id: str
    plan_type: str  # "single_model" | "multi_stage"
    selected_model: Optional[str]
    stage_routes: List[StageRoute]
    fallback_models: List[str]
    verifier_models: List[str]
    expected_latency_ms: float
    expected_cost_usd: float
    expected_quality: float
    confidence: float
    utility: float
    confidence_margin: float
    profile_used: str
    explanation: Dict[str, Any]
    trace: Dict[str, Any]


@dataclass
class RoutingDecision:
    selected_plan: Optional[ExecutionPlan]
    abstain: bool
    escalate_to_human: bool
    decision_record: Dict[str, Any]


LLMParserCallable = Callable[..., "StructuredSemanticParse"]
