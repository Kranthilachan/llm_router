"""
Deterministic + Semantic Parsing.

Splits request understanding into two independent tracks:
  - deterministic_extract(): hard, keyword-based constraint extraction
    (JSON mode, function calling, web search, OCR, citations, verifier).
  - fallback_structured_semantic_parse(): heuristic semantic classifier
    (task family, domain, risk, workflow decomposition). This is the
    default "parser". A real structured-output LLM parser can be plugged
    in via `llm_parser` in parse_task_request()/route() without changing
    any downstream stage.

parse_task_request() merges artifact inspection + deterministic facts +
semantic inference into the single canonical TaskFeatures object that the
rest of the pipeline consumes.
"""

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from app.config import SUPPORTED_FORMATS
from app.core.artifact_inspection import detect_conflicts, inspect_artifacts
from app.core.domain_classifier import (
    LEARNED_DOMAIN_CONFIDENCE_THRESHOLD,
    RISK_METADATA,
    classify_domain,
)
from app.models.schemas import (
    ArtifactProfile,
    RequestConstraints,
    StructuredSemanticParse,
    TaskFeatures,
    TenantContext,
)


def deterministic_extract(prompt: str, input_formats: List[str], estimated_tokens: int) -> Dict[str, Any]:
    p = prompt.lower()
    required_formats = sorted(set(input_formats) | {"text"})
    return {
        "required_formats": required_formats,
        "requires_json": any(k in p for k in ["return json", "respond in json", "json output", "json schema", "structured json"]),
        "requires_function_calling": any(k in p for k in ["tool", "function call", "call a tool", "api call", "use tools"]),
        "requires_web_search": any(k in p for k in ["latest", "today", "current", "search the web", "news", "real-time"]),
        "requires_ocr": (
            any(fmt in input_formats for fmt in ["pdf", "image"])
            and any(k in p for k in ["scan", "extract text", "read the document", "ocr", "handwritten", "digitize"])
        ),
        "requires_citations": any(k in p for k in ["cite", "citations", "sources", "references", "bibliography"]),
        "requires_verifier": any(k in p for k in ["verify", "double-check", "validator", "review", "compliance", "audit"]),
        "min_context_window": estimated_tokens,
    }


def fallback_structured_semantic_parse(
    prompt: str,
    input_formats: List[str],
    estimated_tokens: int,
    artifacts: List[ArtifactProfile],
) -> StructuredSemanticParse:
    """Heuristic parser. Swappable for a real structured LLM parser."""
    p = prompt.lower()
    secondary: List[str] = []
    required_stages: List[str] = []
    workflow_graph: List[Dict[str, Any]] = []
    document_type = "generic"
    reason_parts = []

    # ---- Primary family detection ----
    if any(fmt == "audio" for fmt in input_formats):
        primary = "audio"
        secondary = ["summarization"] if "summarize" in p else ["chat"]
        reason_parts.append("Audio input detected.")
    elif any(fmt == "video" for fmt in input_formats):
        primary = "vision"
        secondary = ["audio", "summarization"]
        reason_parts.append("Video input implies multimodal understanding.")
    elif any(fmt == "image" for fmt in input_formats) and any(k in p for k in ["extract text", "ocr", "scan", "handwritten"]):
        primary = "ocr"
        secondary = ["document_qa"] if any(k in p for k in ["summarize", "analyze", "tabulate"]) else []
        reason_parts.append("Image OCR pattern detected.")
    elif any(fmt == "pdf" for fmt in input_formats):
        if any(k in p for k in ["summarize", "contract", "document", "obligations", "clause", "review"]):
            primary = "document_qa"
            secondary = ["summarization"]
            reason_parts.append("PDF document understanding pattern detected.")
        else:
            primary = "document_qa"
            reason_parts.append("PDF input defaults to document QA.")
    elif any(fmt == "spreadsheet" for fmt in input_formats):
        primary = "document_qa"
        secondary = ["reasoning"]
        reason_parts.append("Spreadsheet analysis pattern detected.")
    elif any(k in p for k in ["python", "bug", "debug", "algorithm", "implement", "code", "sql", "function", "class"]):
        primary = "coding"
        reason_parts.append("Coding intent detected.")
    elif any(k in p for k in ["prove", "integral", "equation", "theorem", "calculate", "derivative"]):
        primary = "mathematics"
        reason_parts.append("Mathematical reasoning detected.")
    elif any(k in p for k in ["translate"]):
        primary = "translation"
        reason_parts.append("Translation intent detected.")
    elif any(k in p for k in ["agent", "orchestrate", "multi-step plan", "workflow", "autonomous"]):
        primary = "agent"
        reason_parts.append("Agentic orchestration intent detected.")
    elif any(k in p for k in ["analyze", "derive", "compare", "reason", "evaluate", "assess"]):
        primary = "reasoning"
        reason_parts.append("General reasoning pattern detected.")
    else:
        primary = "chat"
        reason_parts.append("Defaulting to conversational task.")

    # ---- Domain, risk tier, risk type detection ----
    # Regulated-advice / security keyword branches run FIRST so a statistical
    # classifier can never override policy-critical paths (medical/legal/finance/
    # security -> high risk_tier -> Frontier + verifiers, offensive security ->
    # policy denial). The learned classifier handles business-domain fan-out
    # (crm/hrm/project/accounts) only when none of these keyword rules fire.
    learned_conf = 0.0  # embedding classifier's cosine similarity; 0 if it didn't fire
    if any(k in p for k in ["medical", "diagnosis", "patient", "treatment", "symptom", "chest pain", "clinical"]):
        domain, risk_tier, risk_type = "medical", "high", "regulated_advice"
        document_type = "medical_record"
    elif any(k in p for k in ["legal", "contract", "compliance", "law", "litigation", "obligation", "clause"]):
        domain, risk_tier, risk_type = "legal", "high", "regulated_advice"
        document_type = "contract"
    elif any(k in p for k in ["finance", "tax", "investment", "trading", "portfolio", "stock"]):
        domain, risk_tier, risk_type = "finance", "high", "regulated_advice"
        document_type = "financial_document"
    elif any(k in p for k in ["security", "vulnerability", "exploit", "incident", "breach"]):
        domain, risk_tier, risk_type = "security", "high", "security_sensitive"
        document_type = "security_report"
    else:
        learned_label, learned_conf = classify_domain(prompt)
        if learned_label and learned_conf >= LEARNED_DOMAIN_CONFIDENCE_THRESHOLD and learned_label in RISK_METADATA:
            risk_tier, risk_type, document_type = RISK_METADATA[learned_label]
            domain = learned_label
            reason_parts.append(f"Learned domain classifier: {learned_label} (conf={learned_conf:.2f}).")
        else:
            learned_conf = 0.0
            if "support ticket" in p or "customer" in p or "refund" in p:
                domain, risk_tier, risk_type = "customer_support", "low", "standard"
                document_type = "support_record"
            elif primary in ("coding", "agent"):
                domain, risk_tier, risk_type = "software", "medium", "operational"
            elif primary in ("reasoning", "mathematics"):
                domain, risk_tier, risk_type = "science", "medium", "analytical"
            else:
                domain, risk_tier, risk_type = "general", "low", "standard"

    # ---- Expected output ----
    if any(k in p for k in ["json", "schema", "structured"]):
        expected_output = "structured_json"
    elif primary == "coding":
        expected_output = "code"
    else:
        expected_output = "free_text"

    # ---- Ambiguity ----
    ambiguity_score = 0.15
    if any(k in p for k in ["maybe", "not sure", "either", "somehow", "approximately", "i think"]):
        ambiguity_score = 0.55
    if any(k in p for k in ["unclear", "ambiguous", "open-ended"]):
        ambiguity_score = 0.75

    # ---- Actionability ----
    actionability = "advisory"
    if any(k in p for k in ["do this", "execute", "send", "file", "submit", "trade", "prescribe"]):
        actionability = "high"

    # ---- Decomposition and stages ----
    decomposition_needed = False
    needs_verification = False
    if primary == "ocr":
        required_stages.append("ocr")
        if any(k in p for k in ["tabulate", "summarize", "analyze"]):
            required_stages.append("document_qa")
            decomposition_needed = True
    if primary == "document_qa":
        required_stages.append("document_qa")
        if domain in {"legal", "medical", "finance"}:
            required_stages.append("domain_reasoning")
            decomposition_needed = True
            needs_verification = True
    if expected_output == "structured_json":
        required_stages.append("structured_output")
    if primary == "audio":
        required_stages.append("audio_understanding")
        if "summarize" in p:
            required_stages.append("summarization")
            decomposition_needed = True
    if primary == "vision":
        required_stages.append("vision_understanding")
    if primary == "coding":
        required_stages.append("coding")
    if not required_stages:
        required_stages.append(primary)

    for i, stage in enumerate(required_stages):
        workflow_graph.append({
            "stage_id": i + 1,
            "stage_name": stage,
            "depends_on": [] if i == 0 else [i],
        })

    # If the embedding classifier fired, use its cosine similarity as the
    # parser confidence — this is a real measurement of how close the prompt
    # was to the training set. Otherwise fall back to a keyword-heuristic
    # baseline. Both paths still get dinged for ambiguity / decomposition.
    if learned_conf > 0.0:
        parser_confidence = learned_conf
    else:
        parser_confidence = 0.78
    if ambiguity_score > 0.5:
        parser_confidence -= 0.15
    if decomposition_needed:
        parser_confidence -= 0.05
    parser_confidence = max(0.0, min(1.0, parser_confidence))

    return StructuredSemanticParse(
        primary_family=primary,
        secondary_families=secondary,
        required_stages=required_stages,
        workflow_graph=workflow_graph,
        domain=domain,
        risk_tier=risk_tier,
        risk_type=risk_type,
        expected_output=expected_output,
        ambiguity_score=ambiguity_score,
        actionability=actionability,
        document_type=document_type,
        decomposition_needed=decomposition_needed,
        needs_verification=needs_verification,
        parser_confidence=parser_confidence,
        reason_summary=" ".join(reason_parts),
    )


def validate_structured_parse(data: StructuredSemanticParse) -> StructuredSemanticParse:
    allowed_families = {
        "coding", "reasoning", "mathematics", "chat", "vision",
        "ocr", "document_qa", "summarization", "translation", "agent", "audio"
    }
    if data.primary_family not in allowed_families:
        raise ValueError(f"Invalid primary family: {data.primary_family}")
    if data.risk_tier not in {"low", "medium", "high"}:
        raise ValueError(f"Invalid risk tier: {data.risk_tier}")
    if data.expected_output not in {"structured_json", "free_text", "code"}:
        raise ValueError(f"Invalid expected output: {data.expected_output}")
    if not (0.0 <= data.ambiguity_score <= 1.0):
        raise ValueError("Ambiguity score must be in [0, 1].")
    return data


def parse_with_llm_interface(
    prompt: str,
    input_formats: List[str],
    estimated_tokens: int,
    artifacts: List[ArtifactProfile],
    llm_parser: Optional[Callable] = None,
) -> StructuredSemanticParse:
    """Use real LLM parser if provided, else fall back to heuristic."""
    if llm_parser is not None:
        parsed = llm_parser(
            prompt=prompt,
            input_formats=input_formats,
            estimated_tokens=estimated_tokens,
            artifacts=[asdict(a) for a in artifacts],
        )
        if isinstance(parsed, StructuredSemanticParse):
            return validate_structured_parse(parsed)
        if isinstance(parsed, dict):
            return validate_structured_parse(StructuredSemanticParse(**parsed))
        raise ValueError("llm_parser returned unsupported type.")
    return validate_structured_parse(
        fallback_structured_semantic_parse(prompt, input_formats, estimated_tokens, artifacts)
    )


def infer_workflow_profile(primary_family: str, domain: str, input_formats: List[str], prompt: str) -> str:
    p = prompt.lower()
    if "support" in p or domain == "customer_support":
        return "customer_support_summarization"
    if domain == "legal" and "contract" in p:
        return "contract_review_intake"
    if primary_family == "coding":
        return "coding_assistant"
    if primary_family == "ocr" and any(fmt in input_formats for fmt in ["image", "pdf"]):
        return "invoice_ocr_pipeline"
    if primary_family == "audio":
        return "real_time_voice_agent" if "real-time" in p else "audio_summary"
    if primary_family == "translation":
        return "multilingual_chat"
    if domain == "research":
        return "research_drafting"
    return "generic_balanced"


def parse_task_request(
    prompt: str,
    input_formats: Optional[List[str]] = None,
    estimated_tokens: int = 2000,
    estimated_output_tokens: int = 1200,
    artifact_hints: Optional[List[Dict[str, Any]]] = None,
    llm_parser: Optional[Callable] = None,
    request_constraints: Optional[RequestConstraints] = None,
    tenant_context: Optional[TenantContext] = None,
    files: Optional[List[str]] = None,
) -> TaskFeatures:
    rc = request_constraints or RequestConstraints()
    tc = tenant_context or TenantContext()

    # ---- Determine artifacts and formats ----
    if files:
        artifacts = inspect_artifacts(prompt=prompt, artifact_hints=artifact_hints, files=files)
        input_formats = sorted({a.format for a in artifacts} | {"text"})
    else:
        input_formats = input_formats or ["text"]
        bad = [f for f in input_formats if f not in SUPPORTED_FORMATS]
        if bad:
            raise ValueError(f"Unsupported formats: {bad}. Allowed: {SUPPORTED_FORMATS}")
        artifacts = inspect_artifacts(input_formats=input_formats, prompt=prompt,
                                      artifact_hints=artifact_hints)

    bad = [f for f in input_formats if f not in SUPPORTED_FORMATS]
    if bad:
        raise ValueError(f"Unsupported formats: {bad}. Allowed: {SUPPORTED_FORMATS}")

    # ---- Conflict detection + semantic aggregation ----
    conflict_flags = detect_conflicts(prompt, artifacts)
    previews = [a.extracted_text_preview for a in artifacts if a.extracted_text_preview]
    extracted_text_summary = " ".join(previews)[:2000] if previews else None
    detected_languages = sorted({a.detected_language for a in artifacts if a.detected_language})
    total_file_size = sum(a.file_size_bytes or 0 for a in artifacts)

    hard = deterministic_extract(prompt, input_formats, estimated_tokens)
    if not hard["requires_ocr"]:
        for a in artifacts:
            if a.format in {"pdf", "image"} and (a.scan_likelihood or 0) >= 0.6:
                hard["requires_ocr"] = True
                break

    soft = parse_with_llm_interface(prompt, input_formats, estimated_tokens, artifacts, llm_parser)

    # ---- Topic-driven domain/document override ----
    inferred_topics = [a.inferred_topic for a in artifacts if a.inferred_topic]
    if inferred_topics and soft.domain == "general":
        topic_to_domain = {
            "legal_contract": ("legal", "high", "regulated_advice", "contract"),
            "financial_document": ("finance", "high", "regulated_advice", "financial_document"),
            "medical_record": ("medical", "high", "regulated_advice", "medical_record"),
            "security_report": ("security", "high", "security_sensitive", "security_report"),
            "research_paper": ("research", "low", "standard", "research_paper"),
            "support_record": ("customer_support", "low", "standard", "support_record"),
        }
        topic = inferred_topics[0]
        if topic in topic_to_domain:
            soft.domain, soft.risk_tier, soft.risk_type, soft.document_type = topic_to_domain[topic]

    complexity = "high" if estimated_tokens > 150000 else "medium" if estimated_tokens > 30000 else "low"
    workflow_profile = infer_workflow_profile(soft.primary_family, soft.domain, input_formats, prompt)
    requires_verifier = (
        hard["requires_verifier"]
        or soft.needs_verification
        or soft.risk_tier == "high"
        or rc.mandatory_verifier
    )
    safety_sensitive = soft.risk_tier == "high" or soft.risk_type in {"regulated_advice", "security_sensitive"}
    if rc.no_web_access:
        hard["requires_web_search"] = False

    return TaskFeatures(
        raw_prompt=prompt,
        input_formats=sorted(set(input_formats) | {"text"}),
        estimated_tokens=estimated_tokens,
        estimated_output_tokens=estimated_output_tokens,
        artifacts=artifacts,
        primary_family=soft.primary_family,
        secondary_families=soft.secondary_families,
        required_stages=soft.required_stages,
        workflow_graph=soft.workflow_graph,
        complexity=complexity,
        domain=soft.domain,
        risk_tier=soft.risk_tier,
        risk_type=soft.risk_type,
        required_formats=hard["required_formats"],
        requires_json=hard["requires_json"],
        requires_function_calling=hard["requires_function_calling"],
        requires_web_search=hard["requires_web_search"],
        requires_ocr=hard["requires_ocr"],
        requires_citations=hard["requires_citations"],
        requires_verifier=requires_verifier,
        min_context_window=hard["min_context_window"],
        expected_output=soft.expected_output,
        ambiguity_score=soft.ambiguity_score,
        safety_sensitive=safety_sensitive,
        actionability=soft.actionability,
        document_type=soft.document_type,
        decomposition_needed=soft.decomposition_needed,
        workflow_profile=workflow_profile,
        parser_confidence=soft.parser_confidence,
        conflict_flags=conflict_flags,
        extracted_text_summary=extracted_text_summary,
        detected_languages=detected_languages,
        total_file_size_bytes=total_file_size,
        request_constraints=rc,
        tenant_context=tc,
    )
